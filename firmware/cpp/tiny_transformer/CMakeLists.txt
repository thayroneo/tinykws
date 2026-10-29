cmake_minimum_required(VERSION 3.16)
project(tiny_transformer C)
add_library(tiny_transformer tiny_transformer.c)
target_link_libraries(tiny_transformer m)
