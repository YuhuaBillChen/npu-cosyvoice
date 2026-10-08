# Build inside the image from docker/Dockerfile (XRT runtime + headers, cmake).
XRT_INC ?= /opt/xrt-dev/usr/include
CXXFLAGS ?= -O3 -march=znver4 -ffast-math
LLAMA_CPP ?= third_party/llama.cpp
LLAMA_CPP_REPO ?= https://github.com/ggml-org/llama.cpp.git
# the version the published numbers were measured with
LLAMA_CPP_COMMIT ?= 9adc7f420c37641921b32e326b3d4a538256b878

all: npu_cosyvoice/libdit_npu.so llm

npu_cosyvoice/libdit_npu.so: csrc/dit_npu.cpp
	g++ $(CXXFLAGS) -std=c++20 -fPIC -shared -I$(XRT_INC) $< -o $@ -l:libxrt_coreutil.so.2 -pthread

# CosyVoice3's LM backbone on the CPU through llama.cpp (no GPU backend)
llm: npu_cosyvoice/libcv_llm.so

$(LLAMA_CPP)/include/llama.h:
	git clone --quiet $(LLAMA_CPP_REPO) $(LLAMA_CPP)
	git -C $(LLAMA_CPP) checkout --quiet $(LLAMA_CPP_COMMIT)

$(LLAMA_CPP)/build/bin/libllama.so: $(LLAMA_CPP)/include/llama.h
	cmake -S $(LLAMA_CPP) -B $(LLAMA_CPP)/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
		-DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
		-DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON \
		-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON  # its libraries find each other wherever the checkout is mounted
	cmake --build $(LLAMA_CPP)/build --target llama llama-quantize -j$$(nproc)

npu_cosyvoice/libcv_llm.so: csrc/cv_llm.c $(LLAMA_CPP)/build/bin/libllama.so
	cc -O2 -shared -fPIC $< -I$(LLAMA_CPP)/include -I$(LLAMA_CPP)/ggml/include \
		-L$(LLAMA_CPP)/build/bin -lllama -Wl,-rpath,'$$ORIGIN/../$(LLAMA_CPP)/build/bin' -o $@

clean:
	rm -f npu_cosyvoice/libdit_npu.so npu_cosyvoice/libcv_llm.so
.PHONY: all llm clean
