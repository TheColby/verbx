cmake_minimum_required(VERSION 3.22)

project(verbx_plugin VERSION 0.8.0 LANGUAGES C CXX)

option(VERBX_ENABLE_JUCE_PLUGIN "Build the VERBX AU/AUv3/VST3 plug-in target" OFF)

if(NOT VERBX_ENABLE_JUCE_PLUGIN)
  message(STATUS "VERBX_ENABLE_JUCE_PLUGIN is OFF; JUCE plug-in target is not built.")
  return()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(JUCE CONFIG REQUIRED)

add_subdirectory(../verbx_c verbx_c_core_build)

juce_add_plugin(VERBXPlugin
  COMPANY_NAME "VERBX"
  IS_SYNTH FALSE
  NEEDS_MIDI_INPUT FALSE
  NEEDS_MIDI_OUTPUT FALSE
  IS_MIDI_EFFECT FALSE
  EDITOR_WANTS_KEYBOARD_FOCUS FALSE
  COPY_PLUGIN_AFTER_BUILD FALSE
  PLUGIN_MANUFACTURER_CODE Clby
  PLUGIN_CODE Vrbx
  FORMATS AU AUv3 VST3 Standalone
  PRODUCT_NAME "VERBX"
)

target_sources(VERBXPlugin
  PRIVATE
    src/VerbXPluginProcessor.cpp
    src/VerbXPluginEditor.cpp
)

target_include_directories(VERBXPlugin
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../verbx_c/include
)

target_link_libraries(VERBXPlugin
  PRIVATE
    verbx_c_core
    juce::juce_audio_utils
)

target_compile_definitions(VERBXPlugin
  PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
)
