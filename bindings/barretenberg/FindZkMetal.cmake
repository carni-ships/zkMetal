# FindZkMetal.cmake -- Standard CMake find module for zkMetal
#
# Finds the zkMetal cryptographic library for Apple Silicon GPU acceleration.
#
# Usage:
#   list(APPEND CMAKE_MODULE_PATH "/path/to/this/dir")
#   find_package(ZkMetal)
#
# Imported targets:
#   ZkMetal::ZkMetal   - static library with headers, frameworks, and -DHAS_ZKMETAL=1
#
# Result variables:
#   ZkMetal_FOUND          - TRUE if found
#   ZkMetal_INCLUDE_DIRS   - header search path
#   ZkMetal_LIBRARIES      - libraries to link
#   ZkMetal_VERSION        - version string (if detectable)
#
# Hints:
#   ZKMETAL_ROOT           - CMake variable or environment variable pointing to install prefix

include(FindPackageHandleStandardArgs)

# Only meaningful on Apple platforms
if(NOT APPLE)
    set(ZkMetal_FOUND FALSE)
    return()
endif()

# Collect search paths
set(_ZKMETAL_HINTS "")
if(ZKMETAL_ROOT)
    list(APPEND _ZKMETAL_HINTS "${ZKMETAL_ROOT}")
endif()
if(DEFINED ENV{ZKMETAL_ROOT})
    list(APPEND _ZKMETAL_HINTS "$ENV{ZKMETAL_ROOT}")
endif()

set(_ZKMETAL_PATHS
    /usr/local
    /opt/zkmetal
    /opt/homebrew
)

# --- Header ---
find_path(ZkMetal_INCLUDE_DIR
    NAMES zkmetal.h
    HINTS ${_ZKMETAL_HINTS}
    PATHS ${_ZKMETAL_PATHS}
    PATH_SUFFIXES include
)

# --- Library ---
find_library(ZkMetal_LIBRARY
    NAMES zkmetal libzkmetal
    HINTS ${_ZKMETAL_HINTS}
    PATHS ${_ZKMETAL_PATHS}
    PATH_SUFFIXES lib lib64
)

# --- Version detection from header ---
if(ZkMetal_INCLUDE_DIR AND EXISTS "${ZkMetal_INCLUDE_DIR}/zkmetal.h")
    file(STRINGS "${ZkMetal_INCLUDE_DIR}/zkmetal.h" _ver_major
         REGEX "^#define ZKMETAL_VERSION_MAJOR [0-9]+")
    file(STRINGS "${ZkMetal_INCLUDE_DIR}/zkmetal.h" _ver_minor
         REGEX "^#define ZKMETAL_VERSION_MINOR [0-9]+")
    file(STRINGS "${ZkMetal_INCLUDE_DIR}/zkmetal.h" _ver_patch
         REGEX "^#define ZKMETAL_VERSION_PATCH [0-9]+")

    if(_ver_major AND _ver_minor AND _ver_patch)
        string(REGEX REPLACE ".*MAJOR ([0-9]+)" "\\1" _major "${_ver_major}")
        string(REGEX REPLACE ".*MINOR ([0-9]+)" "\\1" _minor "${_ver_minor}")
        string(REGEX REPLACE ".*PATCH ([0-9]+)" "\\1" _patch "${_ver_patch}")
        set(ZkMetal_VERSION "${_major}.${_minor}.${_patch}")
    endif()
endif()

# --- Standard handling ---
find_package_handle_standard_args(ZkMetal
    REQUIRED_VARS ZkMetal_LIBRARY ZkMetal_INCLUDE_DIR
    VERSION_VAR ZkMetal_VERSION
)

# --- Imported target ---
if(ZkMetal_FOUND AND NOT TARGET ZkMetal::ZkMetal)
    add_library(ZkMetal::ZkMetal STATIC IMPORTED)
    set_target_properties(ZkMetal::ZkMetal PROPERTIES
        IMPORTED_LOCATION "${ZkMetal_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ZkMetal_INCLUDE_DIR}"
        INTERFACE_COMPILE_DEFINITIONS "HAS_ZKMETAL=1"
    )

    # Metal and Foundation frameworks required by libzkmetal.a
    find_library(_ZkMetal_Metal Metal)
    find_library(_ZkMetal_Foundation Foundation)
    if(_ZkMetal_Metal AND _ZkMetal_Foundation)
        set_property(TARGET ZkMetal::ZkMetal APPEND PROPERTY
            INTERFACE_LINK_LIBRARIES
                "${_ZkMetal_Metal}"
                "${_ZkMetal_Foundation}"
        )
    endif()
endif()

# Export convenience variables
set(ZkMetal_INCLUDE_DIRS "${ZkMetal_INCLUDE_DIR}")
set(ZkMetal_LIBRARIES "${ZkMetal_LIBRARY}")

mark_as_advanced(ZkMetal_INCLUDE_DIR ZkMetal_LIBRARY)
