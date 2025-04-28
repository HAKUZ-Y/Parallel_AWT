CXX := /opt/homebrew/opt/llvm/bin/clang++

CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -pedantic -fopenmp \
            -I/opt/homebrew/opt/libomp/include

LDFLAGS := -L/opt/homebrew/opt/libomp/lib

TARGET := awt
SRCS := main.cpp
OBJS := $(SRCS:.cpp=.o)
DEPS := $(SRCS:.cpp=.d)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@


-include $(DEPS)


run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all run clean
