ifeq ($(GSKS_USE_INTEL),true)
include $(GSKS_DIR)/make.intel.inc
else
include $(GSKS_DIR)/make.gnu.inc
endif

TEST_CC_SRC=  \
                 test_dgsks.c \

TEST_CPP_SRC= \
                 test_dgsks_list.cpp \

TEST_EXE= $(TEST_CC_SRC:.c=.x) $(TEST_CPP_SRC:.cpp=.x)

all: $(TEST_EXE)


# ---------------------------------------------------------------------------
# Executable files compiling rules
# ---------------------------------------------------------------------------
%.x: %.c 
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

%.x: %.cpp 
	$(CXX) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)
# ---------------------------------------------------------------------------

clean:
	rm -f $(TEST_EXE)
