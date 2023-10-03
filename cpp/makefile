CXX = g++
CXXFLAGS = -std=c++0x -Wall -pedantic-errors -g

SRCS =  main.cpp num.cpp
OBJS = ${SRCS:.cpp=.o}
HEADERS = num.h

MAIN = cpp_app

all: ${MAIN}
	@echo   Simple compilter named cpp_app has been compiled

${MAIN}: ${OBJS}
	${CXX} ${CXXFLAGS} ${OBJS} -o ${MAIN}

.cpp.o:
	${CXX} ${CXXFLAGS} -c $< -o $@

clean:
	${RM} ${OBJS} ${MAIN} *.o *~.