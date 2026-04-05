# CMakeLists.txt -- zkMetal integration target for Barretenberg
#
# Usage from BB's root CMakeLists.txt:
#   add_subdirectory(path/to/zkMetal/bindings/barretenberg)
#   target_link_libraries(barretenberg PRIVATE ZkMetal::ZkMetal)
#
# Or use FindZkMetal.cmake as a find module (preferred).

cmake_minimum_required(VERSION 3.16)

# Only build on macOS / Apple Silicon
if(NOT APPLE)
    message(STATUS "zkMetal: skipping (not macOS)")
    return()
endif()

# Allow override via ZKMETAL_ROOT env var or cmake variable
if(NOT ZKMETAL_ROOT)
    if(DEFINED ENV{ZKMETAL_ROOT})
        set(ZKMETAL_ROOT "$ENV{ZKMETAL_ROOT}")
    endif()
endif()

# Search paths
set(_ZKMETAL_SEARCH_PATHS
    ${ZKMETAL_ROOT}
    /usr/local
    /opt/zkmetal
    /opt/homebrew
    ${CMAKE_PREFIX_PATH}
)

# Find the header
find_path(ZKMETAL_INCLUDE_DIR
    NAMES zkmetal.h
    PATHS ${_ZKMETAL_SEARCH_PATHS}
    PATH_SUFFIXES include
)

# Find the static library
find_library(ZKMETAL_LIBRARY
    NAMES zkmetal libzkmetal
    PATHS ${_ZKMETAL_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
)

if(ZKMETAL_INCLUDE_DIR AND ZKMETAL_LIBRARY)
    message(STATUS "zkMetal: found ${ZKMETAL_LIBRARY}")
    message(STATUS "zkMetal: headers at ${ZKMETAL_INCLUDE_DIR}")

    # Create imported target
    if(NOT TARGET ZkMetal::ZkMetal)
        add_library(ZkMetal::ZkMetal STATIC IMPORTED)
        set_target_properties(ZkMetal::ZkMetal PROPERTIES
            IMPORTED_LOCATION "${ZKMETAL_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ZKMETAL_INCLUDE_DIR}"
            INTERFACE_COMPILE_DEFINITIONS "HAS_ZKMETAL=1"
        )

        # zkMetal's static lib requires Metal + Foundation frameworks
        find_library(METAL_FRAMEWORK Metal)
        find_library(FOUNDATION_FRAMEWORK Foundation)
        if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
            set_property(TARGET ZkMetal::ZkMetal APPEND PROPERTY
                INTERFACE_LINK_LIBRARIES
                    "${METAL_FRAMEWORK}"
                    "${FOUNDATION_FRAMEWORK}"
            )
        endif()
    endif()

    set(ZkMetal_FOUND TRUE)
else()
    message(STATUS "zkMetal: not found (optional, CPU fallback will be used)")
    set(ZkMetal_FOUND FALSE)
endif()
