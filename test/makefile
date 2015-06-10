include $(KS_DIR)/make.inc

KS_TEST_CC_SRC=  \
                 test_dgsks.c \
								 test_vdexp.c \

KS_TEST_CPP_SRC= \
                 test_dgsks_list.cpp \

KS_TEST_EXE= $(KS_TEST_CC_SRC:.c=.x) $(KS_TEST_CPP_SRC:.cpp=.x)

all: $(KS_TEST_EXE)


# ---------------------------------------------------------------------------
# Executable files compiling rules
# ---------------------------------------------------------------------------
%.x: %.c 
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

%.x: %.cpp 
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)
# ---------------------------------------------------------------------------

clean:
	rm -f $(KS_TEST_EXE)
