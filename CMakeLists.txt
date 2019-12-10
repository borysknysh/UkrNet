cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_C_COMPILER "gcc")

project(Gamapola)
set(CMAKE_BUILD_TYPE "Debug")

ADD_DEFINITIONS( "-Wl,--no-as-needed" )
ADD_DEFINITIONS( "`root-config --cflags`" )
ADD_DEFINITIONS( "-O2 -W -Wall -Wextra -Wno-long-long -Wno-deprecated" )
ADD_DEFINITIONS( "-fno-common" )
ADD_DEFINITIONS( "-fopenmp" )
ADD_DEFINITIONS( "-std=c++14" )

#find_package(root)
FIND_PACKAGE( Boost 1.54 COMPONENTS program_options REQUIRED )
include_directories(${ROOT_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

link_directories($ENV{ROOTSYS}/lib)

file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.h)



add_executable(net net.cxx ${headers})
add_executable(net2 net2.cxx ${headers})

target_link_libraries(net  -lm -lcln -lginac -lstdc++ -fopenmp -pthread -I/usr/local/include/root -L/usr/lib/x86_64-linux-gnu/ -L/usr/aarch64-linux-gnu/lib -L/usr/local/lib/root -L/usr/local/lib/ -lcln -lginac -lTree -lGui -lCore -lCint -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lMinuit2 -lMinuit -lMathMore -lThread -lm -rdynamic -L/usr/local/lib -lboost_filesystem -lboost_system -ldl )

target_link_libraries(net2  -lm -lcln -lginac -lstdc++ -fopenmp -pthread -I/usr/local/include/root -L/usr/lib/x86_64-linux-gnu/ -L/usr/aarch64-linux-gnu/lib -L/usr/local/lib/root -L/usr/local/lib/ -lcln -lginac -lTree -lGui -lCore -lCint -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lMinuit2 -lMinuit -lMathMore -lThread -lm -rdynamic -L/usr/local/lib -lboost_filesystem -lboost_system -ldl )

