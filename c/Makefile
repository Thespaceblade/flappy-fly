CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Wpedantic
LDFLAGS ?= -lm

BIN := flappy_fly
SRCS := flappy.c brain.c main.c
OBJS := $(SRCS:.c=.o)

.PHONY: all clean test

all: $(BIN)

$(BIN): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c flappy.h brain.h
	$(CC) $(CFLAGS) -c -o $@ $<

test: $(BIN)
	./$(BIN) selftest

clean:
	rm -f $(OBJS) $(BIN)
