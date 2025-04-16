CXX := /opt/homebrew/opt/llvm/bin/clang++

CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -pedantic -fopenmp \
            -I/opt/homebrew/opt/libomp/include

LDFLAGS := -L/opt/homebrew/opt/libomp/lib

TARGET := awt_baseline
SRCS := awt_baseline.cpp
OBJS := $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@


run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all run clean
