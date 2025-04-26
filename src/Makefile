# Use mpic++ as the compiler driver
MPICXX := mpic++
CXX := /opt/homebrew/opt/llvm/bin/clang++

# Compilation flags
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -pedantic -fopenmp \
            -I/opt/homebrew/opt/libomp/include

LDFLAGS := -L/opt/homebrew/opt/libomp/lib -lomp

TARGET := awt
SRCS := main.cpp
OBJS := $(SRCS:.cpp=.o)
DEPS := $(SRCS:.cpp=.d)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(MPICXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(MPICXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

# Include generated dependency files
-include $(DEPS)


clean:
	rm -f $(OBJS) $(TARGET) $(DEPS)

.PHONY: all run clean
