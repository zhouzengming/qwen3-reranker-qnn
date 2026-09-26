# Build libqnn_reranker.so natively on the board (Ubuntu aarch64, needs g++ and make).
# include/QNN is copied from your QAIRT SDK by export/make_deploy.py; override with
#   make QNN_INCLUDE=/path/to/qairt/<version>/include/QNN
QNN_INCLUDE ?= include/QNN
CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -fPIC -Wall -Wno-missing-field-initializers

lib/libqnn_reranker.so: src/qnn_reranker.cpp
	@mkdir -p lib
	$(CXX) $(CXXFLAGS) -shared -I$(QNN_INCLUDE) $< -o $@ -ldl

clean:
	rm -f lib/libqnn_reranker.so

.PHONY: clean
