// Qwen3-Reranker on Qualcomm HTP (QNN) — minimal runtime around offline-prepared context binaries.
//
// A model is a sequence of graphs ("parts") executed one after another:
//   single graph : (input_ids int32 [1,L], attention_mask int32 [1,L]) -> logits [1,2], score [1]
//   split model  : (hidden float [1,L,H], attention_mask) -> hidden ... -> logits, score
// The output hidden state of part k feeds part k+1; attention_mask is fed to every part.
// Parts normally come from several single-graph context binaries, each needing ~1 GB of HTP memory at
// L=4096 (~400 MB of it spill-fill scratch). They are registered as one HTP context group sharing a
// spill-fill buffer; when a process domain is full, QNN places further contexts in another PD (seen on
// QCS8550: part 4 lands in pdId 2). createFromBinaryListAsync + shareResources (the Genie path) is
// available via QR_SHARED_LIST=1 but is rejected by the QCS8550 Linux backend of QAIRT 2.50.
// Linking all graphs into one context binary does NOT share scratch (estimate 3.9 GB, does not load).
// Exposed as a small C ABI so it can be driven from Python (ctypes) or C/C++.

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <map>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "HTP/QnnHtpContext.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "QnnInterface.h"
#include "HTP/QnnHtpSystemContext.h"
#include "System/QnnSystemInterface.h"

namespace {

// Entry points resolved with dlsym (signatures from QnnInterface.h / QnnSystemInterface.h).
typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t*** providerList, uint32_t* numProviders);
typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(const QnnSystemInterface_t*** providerList,
                                                                uint32_t* numProviders);

thread_local std::string g_last_error;

void set_error(const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  g_last_error = buf;
}

// ---- Qnn_Tensor_t accessors for the two tensor struct versions ------------------------------
// v1 and v2 share the leading fields we touch, but they live in different union members.
#define QR_TENSOR_FIELD(t, f) ((t).version == QNN_TENSOR_VERSION_2 ? (t).v2.f : (t).v1.f)

const char* tensor_name(const Qnn_Tensor_t& t) { return QR_TENSOR_FIELD(t, name); }
Qnn_DataType_t tensor_dtype(const Qnn_Tensor_t& t) { return QR_TENSOR_FIELD(t, dataType); }
uint32_t tensor_rank(const Qnn_Tensor_t& t) { return QR_TENSOR_FIELD(t, rank); }
const uint32_t* tensor_dims(const Qnn_Tensor_t& t) { return QR_TENSOR_FIELD(t, dimensions); }

void tensor_set_name(Qnn_Tensor_t& t, const char* n) {
  if (t.version == QNN_TENSOR_VERSION_2) t.v2.name = n; else t.v1.name = n;
}
void tensor_set_dims(Qnn_Tensor_t& t, uint32_t* d) {
  if (t.version == QNN_TENSOR_VERSION_2) t.v2.dimensions = d; else t.v1.dimensions = d;
}
void tensor_set_raw_buffer(Qnn_Tensor_t& t, void* data, uint32_t size) {
  Qnn_ClientBuffer_t cb{data, size};
  if (t.version == QNN_TENSOR_VERSION_2) {
    t.v2.memType = QNN_TENSORMEMTYPE_RAW;
    t.v2.clientBuf = cb;
    t.v2.isDynamicDimensions = nullptr;  // static graphs only
  } else {
    t.v1.memType = QNN_TENSORMEMTYPE_RAW;
    t.v1.clientBuf = cb;
  }
}

size_t dtype_size(Qnn_DataType_t dt) {
  switch (dt) {
    case QNN_DATATYPE_FLOAT_32: case QNN_DATATYPE_INT_32: case QNN_DATATYPE_UINT_32: return 4;
    case QNN_DATATYPE_FLOAT_16: case QNN_DATATYPE_INT_16: case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16: case QNN_DATATYPE_SFIXED_POINT_16: return 2;
    case QNN_DATATYPE_INT_64: case QNN_DATATYPE_UINT_64: return 8;
    case QNN_DATATYPE_INT_8: case QNN_DATATYPE_UINT_8: case QNN_DATATYPE_BOOL_8:
    case QNN_DATATYPE_UFIXED_POINT_8: case QNN_DATATYPE_SFIXED_POINT_8: return 1;
    default: return 0;
  }
}

// ---- fp16 <-> fp32 (IEEE 754 half, round-to-nearest-even) -------------------------------------
uint16_t f32_to_f16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
  uint32_t mant = x & 0x7fffffu;
  if (((x >> 23) & 0xff) == 0xff) return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 0x200u : 0));
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    mant |= 0x800000u;
    uint32_t shift = static_cast<uint32_t>(14 - exp);
    uint32_t half = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1);
    uint32_t mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (half & 1))) half++;
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t half = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
  uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1))) half++;
  return static_cast<uint16_t>(half);
}

float f16_to_f32(uint16_t h) {
  uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t mant = h & 0x3ffu;
  uint32_t x;
  if (exp == 0) {
    if (mant == 0) {
      x = sign;
    } else {  // subnormal
      exp = 127 - 15 + 1;
      while (!(mant & 0x400u)) { mant <<= 1; exp--; }
      mant &= 0x3ffu;
      x = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    x = sign | 0x7f800000u | (mant << 13);
  } else {
    x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

// Copy `count` values between host float32/int32 data and a tensor buffer of dtype `dt`.
bool write_float(Qnn_DataType_t dt, void* dst, const float* src, size_t count) {
  if (dt == QNN_DATATYPE_FLOAT_32) { std::memcpy(dst, src, count * 4); return true; }
  if (dt == QNN_DATATYPE_FLOAT_16) {
    auto* d = static_cast<uint16_t*>(dst);
    for (size_t i = 0; i < count; i++) d[i] = f32_to_f16(src[i]);
    return true;
  }
  return false;
}
bool read_float(Qnn_DataType_t dt, const void* src, float* dst, size_t count) {
  if (dt == QNN_DATATYPE_FLOAT_32) { std::memcpy(dst, src, count * 4); return true; }
  if (dt == QNN_DATATYPE_FLOAT_16) {
    auto* s = static_cast<const uint16_t*>(src);
    for (size_t i = 0; i < count; i++) dst[i] = f16_to_f32(s[i]);
    return true;
  }
  return false;
}
bool write_int(Qnn_DataType_t dt, void* dst, const int32_t* src, size_t count) {
  switch (dt) {
    case QNN_DATATYPE_INT_32: case QNN_DATATYPE_UINT_32: std::memcpy(dst, src, count * 4); return true;
    case QNN_DATATYPE_INT_64: case QNN_DATATYPE_UINT_64: {
      auto* d = static_cast<int64_t*>(dst);
      for (size_t i = 0; i < count; i++) d[i] = src[i];
      return true;
    }
    case QNN_DATATYPE_FLOAT_32: {
      auto* d = static_cast<float*>(dst);
      for (size_t i = 0; i < count; i++) d[i] = static_cast<float>(src[i]);
      return true;
    }
    case QNN_DATATYPE_FLOAT_16: {
      auto* d = static_cast<uint16_t*>(dst);
      for (size_t i = 0; i < count; i++) d[i] = f32_to_f16(static_cast<float>(src[i]));
      return true;
    }
    default: return false;
  }
}

bool is_float(Qnn_DataType_t dt) { return dt == QNN_DATATYPE_FLOAT_32 || dt == QNN_DATATYPE_FLOAT_16; }

// ---- per-tensor bookkeeping ------------------------------------------------------------------
struct TensorSlot {
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
  std::string name;
  std::vector<uint32_t> dims;
  std::vector<uint8_t> buffer;
  size_t elements = 0;

  void init_from(const Qnn_Tensor_t& src) {
    tensor = src;  // shallow copy of the metadata struct, then re-point owned fields
    name = tensor_name(src) ? tensor_name(src) : "";
    dims.assign(tensor_dims(src), tensor_dims(src) + tensor_rank(src));
    elements = 1;
    for (uint32_t d : dims) elements *= d;
    buffer.assign(elements * dtype_size(tensor_dtype(src)), 0);
    tensor_set_name(tensor, name.c_str());
    tensor_set_dims(tensor, dims.data());
    tensor_set_raw_buffer(tensor, buffer.data(), static_cast<uint32_t>(buffer.size()));
  }
  Qnn_DataType_t dtype() const { return tensor_dtype(tensor); }
};

struct Part {
  size_t context_index = 0;
  Qnn_GraphHandle_t graph = nullptr;
  std::string graph_name;
  std::vector<TensorSlot> inputs, outputs;
  int main_input = 0, mask_input = 1;  // which input carries ids/hidden and which the mask
  uint64_t spill_fill_bytes = 0;       // HTP spill-fill scratch the graph needs (from binary metadata)
  std::vector<Qnn_Tensor_t> in_array, out_array;
};

struct Context {
  std::string path;
  Qnn_ContextHandle_t handle = nullptr;
  std::vector<Part> graphs;  // graphs found in this binary, in binary order
  uint64_t spill_fill_bytes = 0;
};

struct Runtime {
  void* backend_lib = nullptr;
  void* system_lib = nullptr;
  QNN_INTERFACE_VER_TYPE qnn{};
  QNN_SYSTEM_INTERFACE_VER_TYPE sys{};
  Qnn_LogHandle_t log = nullptr;
  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t device = nullptr;
  uint32_t power_config_id = 0;
  bool power_configured = false;
  bool grouped = false;          // contexts share one spill-fill buffer via group registration
  bool shared_list = false;      // contexts created together with shareResources (list async API)
  std::vector<Context> contexts;
  std::vector<Part*> parts;      // execution order
};

void qnn_log_callback(const char* fmt, QnnLog_Level_t level, uint64_t, va_list args) {
  const char* tag = level == QNN_LOG_LEVEL_ERROR ? "ERROR" : level == QNN_LOG_LEVEL_WARN ? "WARN" : "INFO";
  fprintf(stderr, "[QNN %s] ", tag);
  vfprintf(stderr, fmt, args);
  fprintf(stderr, "\n");
}

bool load_interfaces(Runtime& rt, const char* backend_path, const char* system_path) {
  rt.backend_lib = dlopen(backend_path, RTLD_NOW | RTLD_GLOBAL);
  if (!rt.backend_lib) { set_error("dlopen(%s) failed: %s", backend_path, dlerror()); return false; }
  auto get_providers = reinterpret_cast<QnnInterfaceGetProvidersFn_t>(dlsym(rt.backend_lib, "QnnInterface_getProviders"));
  if (!get_providers) { set_error("QnnInterface_getProviders not found"); return false; }
  const QnnInterface_t** providers = nullptr;
  uint32_t n = 0;
  if (get_providers(&providers, &n) != QNN_SUCCESS || !providers || n == 0) {
    set_error("no QNN interface providers"); return false;
  }
  bool found = false;
  for (uint32_t i = 0; i < n; i++) {
    if (providers[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
        providers[i]->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR) {
      rt.qnn = providers[i]->QNN_INTERFACE_VER_NAME;
      found = true;
      break;
    }
  }
  if (!found) { set_error("backend QNN API version is incompatible with these headers"); return false; }

  rt.system_lib = dlopen(system_path, RTLD_NOW | RTLD_LOCAL);
  if (!rt.system_lib) { set_error("dlopen(%s) failed: %s", system_path, dlerror()); return false; }
  auto get_sys = reinterpret_cast<QnnSystemInterfaceGetProvidersFn_t>(dlsym(rt.system_lib, "QnnSystemInterface_getProviders"));
  if (!get_sys) { set_error("QnnSystemInterface_getProviders not found"); return false; }
  const QnnSystemInterface_t** sys_providers = nullptr;
  if (get_sys(&sys_providers, &n) != QNN_SUCCESS || !sys_providers || n == 0) {
    set_error("no QNN system interface providers"); return false;
  }
  found = false;
  for (uint32_t i = 0; i < n; i++) {
    if (sys_providers[i]->systemApiVersion.major == QNN_SYSTEM_API_VERSION_MAJOR &&
        sys_providers[i]->systemApiVersion.minor >= QNN_SYSTEM_API_VERSION_MINOR) {
      rt.sys = sys_providers[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
      found = true;
      break;
    }
  }
  if (!found) { set_error("system library API version is incompatible with these headers"); return false; }
  return true;
}

// Vote for maximum HTP clocks ("burst"), as qnn-net-run --perf_profile burst does.
void configure_burst(Runtime& rt) {
  if (!rt.qnn.deviceGetInfrastructure) return;
  QnnDevice_Infrastructure_t infra = nullptr;
  if (rt.qnn.deviceGetInfrastructure(&infra) != QNN_SUCCESS || !infra) return;
  auto* htp = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(infra);
  if (htp->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) return;
  QnnHtpDevice_PerfInfrastructure_t perf = htp->perfInfra;
  if (!perf.createPowerConfigId || !perf.setPowerConfig) return;
  if (perf.createPowerConfigId(0, 0, &rt.power_config_id) != QNN_SUCCESS) return;

  QnnHtpPerfInfrastructure_PowerConfig_t dcvs;
  std::memset(&dcvs, 0, sizeof(dcvs));
  dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  dcvs.dcvsV3Config.contextId = rt.power_config_id;
  dcvs.dcvsV3Config.setDcvsEnable = 1;
  dcvs.dcvsV3Config.dcvsEnable = 0;
  dcvs.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  dcvs.dcvsV3Config.setSleepLatency = 1;
  dcvs.dcvsV3Config.sleepLatency = 40;
  dcvs.dcvsV3Config.setSleepDisable = 1;
  dcvs.dcvsV3Config.sleepDisable = 1;
  dcvs.dcvsV3Config.setBusParams = 1;
  dcvs.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.setCoreParams = 1;
  dcvs.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

  QnnHtpPerfInfrastructure_PowerConfig_t latency;
  std::memset(&latency, 0, sizeof(latency));
  latency.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
  latency.rpcControlLatencyConfig = 100;

  QnnHtpPerfInfrastructure_PowerConfig_t polling;
  std::memset(&polling, 0, sizeof(polling));
  polling.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
  polling.rpcPollingTimeConfig = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_MAX_RPC_POLLING_TIME;

  const QnnHtpPerfInfrastructure_PowerConfig_t* configs[] = {&dcvs, &latency, &polling, nullptr};
  if (perf.setPowerConfig(rt.power_config_id, configs) == QNN_SUCCESS) {
    rt.power_configured = true;
  } else {
    fprintf(stderr, "[qnn_reranker] warning: could not apply burst power config\n");
  }
}

uint64_t graph_spill_fill(const QnnSystemContext_GraphInfo_t& g) {
  if (g.version != QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3 || !g.graphInfoV3.graphBlobInfo) return 0;
  auto* blob = static_cast<const QnnHtpSystemContext_GraphBlobInfo_t*>(g.graphInfoV3.graphBlobInfo);
  return blob->version == QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_VERSION_V1
             ? blob->contextBinaryGraphBlobInfoV1.spillFillBufferSize : 0;
}

uint64_t context_spill_fill(const QnnSystemContext_BinaryInfo_t* info) {
  const void* hw = nullptr;  // V1/V2 binaries: per-context hardware info blob
  if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) hw = info->contextBinaryInfoV1.hwInfoBlob;
  if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) hw = info->contextBinaryInfoV2.hwInfoBlob;
  auto* hwb = static_cast<const QnnHtpSystemContext_HwBlobInfo_t*>(hw);
  return hwb && hwb->version == QNN_SYSTEM_CONTEXT_HTP_HW_INFO_BLOB_VERSION_V1
             ? hwb->contextBinaryHwInfoBlobV1_t.spillFillBufferSize : 0;
}

// Pull graph names + IO tensor metadata out of a context binary without creating a context.
bool read_binary_metadata(Runtime& rt, const void* data, uint64_t size, Context& ctx) {
  QnnSystemContext_Handle_t sys_ctx = nullptr;
  if (rt.sys.systemContextCreate(&sys_ctx) != QNN_SUCCESS) { set_error("systemContextCreate failed"); return false; }
  const QnnSystemContext_BinaryInfo_t* info = nullptr;
  Qnn_ContextBinarySize_t info_size = 0;
  bool ok = rt.sys.systemContextGetBinaryInfo(sys_ctx, const_cast<void*>(data), size, &info, &info_size) == QNN_SUCCESS && info;
  if (!ok) set_error("systemContextGetBinaryInfo failed for %s", ctx.path.c_str());

  uint32_t num_graphs = 0;
  const QnnSystemContext_GraphInfo_t* graphs = nullptr;
  if (ok) {
    switch (info->version) {
      case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1: num_graphs = info->contextBinaryInfoV1.numGraphs; graphs = info->contextBinaryInfoV1.graphs; break;
      case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2: num_graphs = info->contextBinaryInfoV2.numGraphs; graphs = info->contextBinaryInfoV2.graphs; break;
      case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3: num_graphs = info->contextBinaryInfoV3.numGraphs; graphs = info->contextBinaryInfoV3.graphs; break;
      default: ok = false; set_error("unsupported binary info version %d", static_cast<int>(info->version));
    }
  }
  if (ok && (num_graphs == 0 || !graphs)) { ok = false; set_error("%s: no graphs in binary", ctx.path.c_str()); }
  if (ok) {
    ctx.graphs.resize(num_graphs);  // sized once: TensorSlot keeps pointers into its own storage
    uint64_t ctx_spill = context_spill_fill(info);
    for (uint32_t gi = 0; gi < num_graphs && ok; gi++) {
      const auto& g = graphs[gi];
      const char* name = nullptr;
      const Qnn_Tensor_t *in = nullptr, *out = nullptr;
      uint32_t n_in = 0, n_out = 0;
      if (g.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
        name = g.graphInfoV1.graphName; in = g.graphInfoV1.graphInputs; n_in = g.graphInfoV1.numGraphInputs;
        out = g.graphInfoV1.graphOutputs; n_out = g.graphInfoV1.numGraphOutputs;
      } else if (g.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
        name = g.graphInfoV2.graphName; in = g.graphInfoV2.graphInputs; n_in = g.graphInfoV2.numGraphInputs;
        out = g.graphInfoV2.graphOutputs; n_out = g.graphInfoV2.numGraphOutputs;
      } else if (g.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
        name = g.graphInfoV3.graphName; in = g.graphInfoV3.graphInputs; n_in = g.graphInfoV3.numGraphInputs;
        out = g.graphInfoV3.graphOutputs; n_out = g.graphInfoV3.numGraphOutputs;
      } else {
        ok = false; set_error("unsupported graph info version %d", static_cast<int>(g.version)); break;
      }
      Part& part = ctx.graphs[gi];
      part.graph_name = name ? name : "";
      part.spill_fill_bytes = graph_spill_fill(g);
      if (part.spill_fill_bytes == 0) part.spill_fill_bytes = ctx_spill;
      ctx.spill_fill_bytes = std::max(ctx.spill_fill_bytes, part.spill_fill_bytes);
      part.inputs.resize(n_in);
      part.outputs.resize(n_out);
      for (uint32_t i = 0; i < n_in; i++) part.inputs[i].init_from(in[i]);
      for (uint32_t i = 0; i < n_out; i++) part.outputs[i].init_from(out[i]);
    }
  }
  rt.sys.systemContextFree(sys_ctx);
  return ok;
}

// Identify the mask input: named *mask*, otherwise the second input.
void resolve_inputs(Part& p) {
  if (p.inputs.size() < 2) return;
  for (size_t i = 0; i < p.inputs.size(); i++) {
    if (p.inputs[i].name.find("mask") != std::string::npos) {
      p.mask_input = static_cast<int>(i);
      p.main_input = i == 0 ? 1 : 0;
      return;
    }
  }
  p.main_input = 0;
  p.mask_input = 1;
}

struct MappedFile {
  void* data = MAP_FAILED;
  uint64_t size = 0;
  explicit MappedFile(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return;
    struct stat st{};
    if (fstat(fd, &st) == 0) {
      size = static_cast<uint64_t>(st.st_size);
      data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    }
    close(fd);
  }
  ~MappedFile() { if (data != MAP_FAILED) munmap(data, size); }
  bool ok() const { return data != MAP_FAILED; }
};

bool read_context_metadata(Runtime& rt, Context& ctx) {
  MappedFile f(ctx.path);
  if (!f.ok()) { set_error("cannot open/mmap %s", ctx.path.c_str()); return false; }
  return read_binary_metadata(rt, f.data, f.size, ctx);
}

Qnn_ErrorHandle_t create_context(Runtime& rt, Context& ctx, const MappedFile& f, Qnn_ContextHandle_t group_first,
                                 uint64_t spill) {
  QnnHtpContext_CustomConfig_t group_cfg;
  std::memset(&group_cfg, 0, sizeof(group_cfg));
  group_cfg.option = QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS;
  group_cfg.groupRegistration.firstGroupHandle = group_first;
  group_cfg.groupRegistration.maxSpillFillBuffer = spill;
  QnnContext_Config_t cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
  cfg.customConfig = &group_cfg;
  const QnnContext_Config_t* cfgs[] = {&cfg, nullptr};
  return rt.qnn.contextCreateFromBinary(rt.backend, rt.device, spill ? cfgs : nullptr, f.data, f.size, &ctx.handle, nullptr);
}

// Create the context of `ctx`. With several contexts, try to join a shared spill-fill group first
// and fall back to a standalone context if the backend rejects the group option.
bool load_context(Runtime& rt, Context& ctx, Qnn_ContextHandle_t group_first, uint64_t spill, int log_level) {
  MappedFile f(ctx.path);
  if (!f.ok()) { set_error("cannot open/mmap %s", ctx.path.c_str()); return false; }
  Qnn_ErrorHandle_t err = QNN_SUCCESS;
  if (spill && (rt.grouped || !group_first)) {
    err = create_context(rt, ctx, f, group_first, spill);
    if (err == QNN_SUCCESS) {
      rt.grouped = true;
    } else if (!group_first) {  // first context: find out whether grouping is supported at all
      if (log_level >= 1)
        fprintf(stderr, "[qnn_reranker] warning: shared spill-fill group not supported (err %lu); "
                        "loading contexts standalone\n", static_cast<unsigned long>(err));
      ctx.handle = nullptr;
      err = create_context(rt, ctx, f, nullptr, 0);
    }
  } else {
    err = create_context(rt, ctx, f, nullptr, 0);
  }
  if (err != QNN_SUCCESS) {
    set_error("contextCreateFromBinary(%s) failed: %lu%s", ctx.path.c_str(), static_cast<unsigned long>(err),
              (group_first && !rt.grouped) ? " (contexts not sharing memory: the HTP process domain is "
                                             "probably exhausted, see README)" : "");
    return false;
  }
  for (auto& part : ctx.graphs) {
    if (rt.qnn.graphRetrieve(ctx.handle, part.graph_name.c_str(), &part.graph) != QNN_SUCCESS) {
      set_error("graphRetrieve(%s) failed", part.graph_name.c_str());
      return false;
    }
    resolve_inputs(part);
    for (auto& t : part.inputs) part.in_array.push_back(t.tensor);
    for (auto& t : part.outputs) part.out_array.push_back(t.tensor);
  }
  return true;
}

// ---- createFromBinaryListAsync with shared resources (preferred for several binaries) ----------
struct ListNotify {
  Runtime* rt;
  size_t ctx_index;
};

void list_notify(Qnn_ContextHandle_t context, Qnn_GraphHandle_t graph, const char* graph_name,
                 QnnContext_createFromBinaryAsyncNotifyType_t type, void* param, Qnn_ErrorHandle_t status) {
  auto* n = static_cast<ListNotify*>(param);
  Context& ctx = n->rt->contexts[n->ctx_index];
  if (status != QNN_SUCCESS) return;
  if (type == QNN_CONTEXT_NOTIFY_TYPE_CONTEXT_INIT) {
    ctx.handle = context;
  } else if (type == QNN_CONTEXT_NOTIFY_TYPE_GRAPH_INIT && graph_name) {
    for (auto& p : ctx.graphs)
      if (p.graph_name == graph_name) p.graph = graph;
    if (!ctx.handle) ctx.handle = context;
  }
}

// Returns QNN_SUCCESS when all contexts were created; on failure nothing stays allocated.
Qnn_ErrorHandle_t load_contexts_shared(Runtime& rt, int log_level) {
  if (!rt.qnn.contextCreateFromBinaryListAsync) return QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE;
  const size_t n = rt.contexts.size();
  std::vector<std::unique_ptr<MappedFile>> files;
  std::vector<ListNotify> notify(n);
  std::vector<std::unique_ptr<QnnContext_Params_t>> params;  // aggregate-initialized: v1 has a const member
  std::vector<const QnnContext_Params_t*> param_ptrs;
  for (size_t i = 0; i < n; i++) {
    files.emplace_back(new MappedFile(rt.contexts[i].path));
    if (!files.back()->ok()) { set_error("cannot open/mmap %s", rt.contexts[i].path.c_str()); return QNN_CONTEXT_ERROR_INVALID_ARGUMENT; }
    notify[i] = ListNotify{&rt, i};
    params.emplace_back(new QnnContext_Params_t{
        QNN_CONTEXT_PARAMS_VERSION_1,
        {QnnContext_ParamsV1_t{nullptr, files.back()->data, files.back()->size, nullptr, list_notify, &notify[i]}}});
    param_ptrs.push_back(params.back().get());
  }
  param_ptrs.push_back(nullptr);

  QnnHtpContext_CustomConfig_t share, opt;
  std::memset(&share, 0, sizeof(share));
  share.option = QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES;
  share.shareResources = true;
  std::memset(&opt, 0, sizeof(opt));
  opt.option = QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES_OPTIMIZATION_TYPE;
  opt.shareResOptType = SEQUENTIAL_WITH_VA_OPTIMIZATION;  // parts run strictly one after another
  QnnContext_Config_t c1, c2;
  std::memset(&c1, 0, sizeof(c1));
  c1.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
  c1.customConfig = &share;
  std::memset(&c2, 0, sizeof(c2));
  c2.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
  c2.customConfig = &opt;
  const QnnContext_Config_t* list_cfg[] = {&c1, &c2, nullptr};

  Qnn_ErrorHandle_t err = rt.qnn.contextCreateFromBinaryListAsync(rt.backend, rt.device, param_ptrs.data(),
                                                                  list_cfg, nullptr);
  bool complete = err == QNN_SUCCESS;
  for (auto& ctx : rt.contexts) {
    if (!ctx.handle) complete = false;
    for (auto& p : ctx.graphs) {  // graph handles not delivered by the callback: retrieve them
      if (complete && !p.graph && rt.qnn.graphRetrieve(ctx.handle, p.graph_name.c_str(), &p.graph) != QNN_SUCCESS)
        complete = false;
    }
  }
  if (!complete) {
    for (auto it = rt.contexts.rbegin(); it != rt.contexts.rend(); ++it) {
      if (it->handle && rt.qnn.contextFree) rt.qnn.contextFree(it->handle, nullptr);
      it->handle = nullptr;
      for (auto& p : it->graphs) p.graph = nullptr;
    }
    if (log_level >= 1)
      fprintf(stderr, "[qnn_reranker] warning: createFromBinaryListAsync with shared resources failed (err %lu)\n",
              static_cast<unsigned long>(err));
    return err == QNN_SUCCESS ? static_cast<Qnn_ErrorHandle_t>(QNN_CONTEXT_ERROR_CREATE_FROM_BINARY) : err;
  }
  for (auto& ctx : rt.contexts) {
    for (auto& part : ctx.graphs) {
      resolve_inputs(part);
      for (auto& t : part.inputs) part.in_array.push_back(t.tensor);
      for (auto& t : part.outputs) part.out_array.push_back(t.tensor);
    }
  }
  rt.shared_list = true;
  if (log_level >= 2) fprintf(stderr, "[qnn_reranker] %zu contexts created with shared resources\n", n);
  return QNN_SUCCESS;
}

void destroy(Runtime* rt) {
  if (!rt) return;
  // grouped contexts: release the members before the first context that owns the shared buffer
  for (auto it = rt->contexts.rbegin(); it != rt->contexts.rend(); ++it) {
    if (it->handle && rt->qnn.contextFree) rt->qnn.contextFree(it->handle, nullptr);
  }
  if (rt->power_configured) {
    QnnDevice_Infrastructure_t infra = nullptr;
    if (rt->qnn.deviceGetInfrastructure && rt->qnn.deviceGetInfrastructure(&infra) == QNN_SUCCESS && infra) {
      auto* htp = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(infra);
      if (htp->perfInfra.destroyPowerConfigId) htp->perfInfra.destroyPowerConfigId(rt->power_config_id);
    }
  }
  if (rt->device && rt->qnn.deviceFree) rt->qnn.deviceFree(rt->device);
  if (rt->backend && rt->qnn.backendFree) rt->qnn.backendFree(rt->backend);
  if (rt->log && rt->qnn.logFree) rt->qnn.logFree(rt->log);
  if (rt->system_lib) dlclose(rt->system_lib);
  // the backend library is intentionally not dlclose'd: HTP keeps worker threads around
  delete rt;
}

}  // namespace

extern "C" {

const char* qr_last_error() { return g_last_error.c_str(); }

// ctx_paths  : context binaries, in execution order
// graph_order: optional list of n_order graph names giving the execution order across all binaries
//              (needed when one binary holds several graphs whose stored order is not the run order);
//              nullptr / 0 = binaries in the given order, graphs in their stored order
// perf_mode  : 1 = burst clocks (recommended), 0 = leave DCVS defaults
// log_level  : 0 = off, 1 = error, 2 = warn, 3 = info
void* qr_create2(const char* backend_path, const char* system_path, const char** ctx_paths, int n_ctx,
                 const char** graph_order, int n_order, int perf_mode, int log_level) {
  auto* rt = new Runtime();
  if (!load_interfaces(*rt, backend_path, system_path)) { destroy(rt); return nullptr; }
  if (log_level > 0 && rt->qnn.logCreate) {
    auto lvl = log_level >= 3 ? QNN_LOG_LEVEL_INFO : log_level == 2 ? QNN_LOG_LEVEL_WARN : QNN_LOG_LEVEL_ERROR;
    rt->qnn.logCreate(qnn_log_callback, lvl, &rt->log);
  }
  if (rt->qnn.backendCreate(rt->log, nullptr, &rt->backend) != QNN_SUCCESS) {
    set_error("backendCreate failed"); destroy(rt); return nullptr;
  }
  if (rt->qnn.deviceCreate &&
      rt->qnn.deviceCreate(rt->log, nullptr, &rt->device) != QNN_SUCCESS) {
    set_error("deviceCreate failed (check the QNN log above; an 'Unsupported SoC model' error means the "
              "runtime libraries do not support this SoC - QCS8550 needs the aarch64-oe-linux-gcc11.2 build)");
    destroy(rt); return nullptr;
  }
  if (perf_mode == 1) configure_burst(*rt);

  rt->contexts.resize(n_ctx);  // sized once: parts are referenced by pointer below
  uint64_t spill = 0;
  for (int i = 0; i < n_ctx; i++) {
    rt->contexts[i].path = ctx_paths[i];
    if (!read_context_metadata(*rt, rt->contexts[i])) { destroy(rt); return nullptr; }
    spill = std::max(spill, rt->contexts[i].spill_fill_bytes);
  }
  if (log_level >= 2)
    fprintf(stderr, "[qnn_reranker] %d context(s), max spill-fill %.1f MiB\n", n_ctx, spill / 1048576.0);
  bool loaded = false;
  // createFromBinaryListAsync + shareResources is opt-in (QR_SHARED_LIST=1): the QCS8550 Linux HTP backend
  // of QAIRT 2.50 rejects it ("Backend does not support shared resources enabled optimization").
  if (n_ctx > 1 && std::getenv("QR_SHARED_LIST")) loaded = load_contexts_shared(*rt, log_level) == QNN_SUCCESS;
  for (int i = 0; !loaded && i < n_ctx; i++) {
    Qnn_ContextHandle_t first = i == 0 ? nullptr : rt->contexts[0].handle;
    if (!load_context(*rt, rt->contexts[i], first, n_ctx > 1 ? spill : 0, log_level)) { destroy(rt); return nullptr; }
  }

  if (graph_order && n_order > 0) {
    for (int k = 0; k < n_order; k++) {
      Part* found = nullptr;
      for (auto& c : rt->contexts)
        for (auto& p : c.graphs)
          if (p.graph_name == graph_order[k]) found = &p;
      if (!found) { set_error("graph '%s' not found in the context binaries", graph_order[k]); destroy(rt); return nullptr; }
      rt->parts.push_back(found);
    }
  } else {
    for (auto& c : rt->contexts)
      for (auto& p : c.graphs) rt->parts.push_back(&p);
  }
  return rt;
}

void* qr_create(const char* backend_path, const char* system_path, const char** ctx_paths, int n_ctx,
                int perf_mode, int log_level) {
  return qr_create2(backend_path, system_path, ctx_paths, n_ctx, nullptr, 0, perf_mode, log_level);
}

void qr_destroy(void* h) { destroy(static_cast<Runtime*>(h)); }

int qr_num_parts(void* h) { return static_cast<int>(static_cast<Runtime*>(h)->parts.size()); }

// How the contexts share memory: 2 = createFromBinaryListAsync shareResources, 1 = spill-fill group
// registration, 0 = standalone contexts.
int qr_contexts_grouped(void* h) {
  auto* rt = static_cast<Runtime*>(h);
  return rt->shared_list ? 2 : rt->grouped ? 1 : 0;
}

// Graph name of part `part` (execution order).
const char* qr_graph_name(void* h, int part) {
  auto* rt = static_cast<Runtime*>(h);
  if (part < 0 || part >= static_cast<int>(rt->parts.size())) return "";
  return rt->parts[part]->graph_name.c_str();
}

// Spill-fill scratch requirement of one part, in bytes (as recorded in its context binary).
unsigned long long qr_spill_fill_bytes(void* h, int part) {
  auto* rt = static_cast<Runtime*>(h);
  if (part < 0 || part >= static_cast<int>(rt->parts.size())) return 0;
  return rt->parts[part]->spill_fill_bytes;
}

int qr_num_tensors(void* h, int part, int is_output) {
  auto* rt = static_cast<Runtime*>(h);
  if (part < 0 || part >= static_cast<int>(rt->parts.size())) return -1;
  return static_cast<int>(is_output ? rt->parts[part]->outputs.size() : rt->parts[part]->inputs.size());
}

// Describe one IO tensor. Returns 0 on success. dtype is the raw Qnn_DataType_t value.
int qr_tensor_info(void* h, int part, int is_output, int index, char* name, int name_len, int* dtype,
                   uint32_t* dims, int* rank) {
  auto* rt = static_cast<Runtime*>(h);
  if (part < 0 || part >= static_cast<int>(rt->parts.size())) return -1;
  auto& v = is_output ? rt->parts[part]->outputs : rt->parts[part]->inputs;
  if (index < 0 || index >= static_cast<int>(v.size())) return -1;
  const auto& t = v[index];
  if (name && name_len > 0) snprintf(name, name_len, "%s", t.name.c_str());
  if (dtype) *dtype = static_cast<int>(t.dtype());
  if (rank) *rank = static_cast<int>(t.dims.size());
  if (dims) for (size_t i = 0; i < t.dims.size(); i++) dims[i] = t.dims[i];
  return 0;
}

// Copy output `index` of `part` (from the last qr_run/qr_run_parts) as float32. Returns elements copied.
long qr_copy_output(void* h, int part, int index, float* dst, long capacity) {
  auto* rt = static_cast<Runtime*>(h);
  if (part < 0 || part >= static_cast<int>(rt->parts.size())) return -1;
  auto& outs = rt->parts[part]->outputs;
  if (index < 0 || index >= static_cast<int>(outs.size())) return -1;
  auto& o = outs[index];
  if (static_cast<long>(o.elements) > capacity) return -1;
  return read_float(o.dtype(), o.buffer.data(), dst, o.elements) ? static_cast<long>(o.elements) : -1;
}

// Execute only the first `num_parts` parts (debugging: inspect intermediate outputs with qr_copy_output).
//   input_ids : int32 [L]      used when part 0 takes token ids (single-graph models)
//   hidden    : float32 [L*H]  used when part 0 takes hidden states (embedding done on the host)
//   mask      : int32 [L]
//   part_ms   : optional double [num_parts] out: per-part execute time
int qr_run_parts(void* h, const int32_t* input_ids, const float* hidden, const int32_t* mask, int num_parts,
                 double* part_ms) {
  auto* rt = static_cast<Runtime*>(h);
  size_t n = std::min(static_cast<size_t>(num_parts), rt->parts.size());
  for (size_t k = 0; k < n; k++) {
    Part& p = *rt->parts[k];
    TensorSlot& main_in = p.inputs[p.main_input];
    TensorSlot& mask_in = p.inputs[p.mask_input];
    if (!write_int(mask_in.dtype(), mask_in.buffer.data(), mask, mask_in.elements)) {
      set_error("part %zu: unsupported mask dtype %d", k, static_cast<int>(mask_in.dtype())); return -1;
    }
    if (k == 0) {
      bool ok = is_float(main_in.dtype())
                    ? (hidden && write_float(main_in.dtype(), main_in.buffer.data(), hidden, main_in.elements))
                    : (input_ids && write_int(main_in.dtype(), main_in.buffer.data(), input_ids, main_in.elements));
      if (!ok) { set_error("part 0: missing or unsupported main input (dtype %d)", static_cast<int>(main_in.dtype())); return -1; }
    } else {
      TensorSlot& prev = rt->parts[k - 1]->outputs[0];
      if (prev.dtype() == main_in.dtype() && prev.buffer.size() == main_in.buffer.size()) {
        std::memcpy(main_in.buffer.data(), prev.buffer.data(), prev.buffer.size());
      } else {  // dtype mismatch between parts: go through float32
        std::vector<float> tmp(prev.elements);
        if (!read_float(prev.dtype(), prev.buffer.data(), tmp.data(), prev.elements) ||
            !write_float(main_in.dtype(), main_in.buffer.data(), tmp.data(), main_in.elements)) {
          set_error("part %zu: cannot convert hidden state between dtypes", k); return -1;
        }
      }
    }
    auto t0 = std::chrono::steady_clock::now();
    Qnn_ErrorHandle_t err = rt->qnn.graphExecute(p.graph, p.in_array.data(), static_cast<uint32_t>(p.in_array.size()),
                                                 p.out_array.data(), static_cast<uint32_t>(p.out_array.size()), nullptr, nullptr);
    auto t1 = std::chrono::steady_clock::now();
    if (err != QNN_GRAPH_NO_ERROR) { set_error("graphExecute failed on part %zu: %lu", k, static_cast<unsigned long>(err)); return -1; }
    if (part_ms) part_ms[k] = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  return 0;
}

// Run all parts for one (already tokenized, left-padded) sample; logits out: float32 [2] = [no, yes].
int qr_run(void* h, const int32_t* input_ids, const float* hidden, const int32_t* mask, float* logits, double* part_ms) {
  auto* rt = static_cast<Runtime*>(h);
  if (qr_run_parts(h, input_ids, hidden, mask, static_cast<int>(rt->parts.size()), part_ms) != 0) return -1;
  for (auto& o : rt->parts.back()->outputs) {  // the 2-element output of the last part is the logits
    if (o.elements == 2) {
      if (!read_float(o.dtype(), o.buffer.data(), logits, 2)) { set_error("unsupported logits dtype"); return -1; }
      return 0;
    }
  }
  set_error("last part has no 2-element logits output");
  return -1;
}

}  // extern "C"
