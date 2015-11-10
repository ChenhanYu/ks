ifeq ($(GSKS_USE_INTEL),true)
include $(GSKS_DIR)/make.intel.inc
else
include $(GSKS_DIR)/make.gnu.inc
endif

FRAME_CC_SRC=     \
								  frame/dgsks.c \
								  frame/dgsks_ref.c \
									frame/ks_util.c \

FRAME_CPP_SRC=    \
								  frame/omp_dgsks_list.cpp \

KERNEL_SRC=       \
								  micro_kernel/$(GSKS_ARCH)/ks_gaussian_int_d8x4.c \
								  micro_kernel/$(GSKS_ARCH)/ks_gaussian_svml_d8x4.c \
								  \
								  micro_kernel/$(GSKS_ARCH)/ks_variable_bandwidth_gaussian_int_d8x4.c \
								  \
								  micro_kernel/$(GSKS_ARCH)/ks_polynomial_int_d8x4.c \
									\
								  micro_kernel/$(GSKS_ARCH)/ks_laplace3d_int_d8x4.c \
								  \
								  micro_kernel/$(GSKS_ARCH)/ks_tanh_int_d8x4.c \
									\
								  micro_kernel/$(GSKS_ARCH)/ks_quartic_int_d8x4.c \
									\
								  micro_kernel/$(GSKS_ARCH)/ks_epanechnikov_int_d8x4.c \
									\
								  micro_kernel/$(GSKS_ARCH)/ks_multiquadratic_int_d8x4.c \
									\
								  micro_kernel/$(GSKS_ARCH)/ks_rank_k_int_d8x4.c \
								  micro_kernel/$(GSKS_ARCH)/ks_rank_k_asm_d8x4.c \
	
FRAME_MIC_CC_SRC= \
								  frame/dgsks_mic.c \
								  frame/dgsks_ref_mic.c \

FRAME_MIC_CPP_SRC=\
								  frame/omp_dgsks_list_mic.cpp \


KERNEL_MIC_SRC=   \
								  micro_kernel/mic/ks_gaussian_int_d16x8.c \
								  micro_kernel/mic/ks_gaussian_int_d16x14.c \
								  micro_kernel/mic/ks_gaussian_asm_d24x8.c \
								  micro_kernel/mic/ks_gaussian_asm_d8x30.c \
								  \
								  micro_kernel/mic/ks_rank_k_int_d16x14.c \
								  micro_kernel/mic/ks_rank_k_asm_d30x8.c \
								  micro_kernel/mic/ks_rank_k_asm_d8x30.c \
								  \
								  micro_kernel/mic/ks_gaussian_int_d16x14_var2.c \


GSKS_OBJ=$(FRAME_CC_SRC:.c=.o) $(FRAME_CPP_SRC:.cpp=.o) $(KERNEL_SRC:.c=.o)

GSKS_MIC_OBJ=$(FRAME_MIC_CC_SRC:.c=.mic) $(FRAME_MIC_CPP_SRC:.cpp=.mic) $(KERNEL_MIC_SRC:.c=.mic)

all: $(LIBGSKS) $(SHAREDLIBGSKS) TESTGSKS

TESTGSKS: $(LIBGSKS)
	cd $(GSKS_DIR)/test && $(MAKE) && cd $(GSKS_DIR)


$(LIBGSKSMIC): $(GSKS_MIC_OBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(GSKS_MIC_OBJ) 
	$(RANLIB) $@


$(LIBGSKS): $(GSKS_OBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(GSKS_OBJ)
	$(RANLIB) $@

$(SHAREDLIBGSKS): $(GSKS_OBJ)
	$(CXX) $(CFLAGS) -shared -o $@ $(GSKS_OBJ) $(LDLIBS)

# ---------------------------------------------------------------------------
# Object files compiling rules
# ---------------------------------------------------------------------------
%.o: %.c 
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ $(LDFLAGS)

%.mic: %.c 
	$(CC) $(CFLAGSMIC) -c $< -o $@ $(LDFLAGS)

%.mic: %.cpp
	$(CXX) $(CFLAGSMIC) -c $< -o $@ $(LDFLAGS)
# ---------------------------------------------------------------------------

clean:
	rm $(GSKS_OBJ) $(LIBGSKS)
	cd $(GSKS_DIR)/test && $(MAKE) clean && cd $(GSKS_DIR)
