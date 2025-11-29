# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# examples

# examples
if (EMSCRIPTEN)
else()
    add_subdirectory(random-trimming)
	add_subdirectory(command-inference)
endif()
