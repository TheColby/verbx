cmake_minimum_required(VERSION 3.22)

if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
  if(NOT CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING
      "macOS plug-in architectures")
  endif()
  if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING
      "Oldest supported macOS release")
  endif()
endif()

project(verbx_plugin VERSION 0.8.0 LANGUAGES C CXX)

option(VERBX_ENABLE_JUCE_PLUGIN "Build the VERBX AU/AUv3/VST3 plug-in target" OFF)
option(VERBX_BUILD_HOST_SMOKE_TEST "Build the JUCE plug-in host smoke test" OFF)
set(VERBX_JUCE_SOURCE_DIR "" CACHE PATH "Optional path to a JUCE source checkout")

if(NOT VERBX_ENABLE_JUCE_PLUGIN)
  message(STATUS "VERBX_ENABLE_JUCE_PLUGIN is OFF; JUCE plug-in target is not built.")
  return()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(VERBX_JUCE_SOURCE_DIR)
  add_subdirectory("${VERBX_JUCE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/JUCE")
else()
  find_package(JUCE CONFIG REQUIRED)
endif()

# JUCE limits AUv3 to the Xcode generator because only Xcode embeds app
# extensions automatically. The wrapper itself builds correctly with standard
# macOS generators; VERBX supplies the missing container embed step below.
if(APPLE AND NOT CMAKE_GENERATOR STREQUAL "Xcode")
  if(NOT TARGET juce_audio_plugin_client_AUv3)
    get_target_property(verbx_juce_plugin_client_path
      juce::juce_audio_plugin_client INTERFACE_JUCE_MODULE_PATH)
    get_filename_component(verbx_juce_modules_path
      "${verbx_juce_plugin_client_path}" DIRECTORY)
    _juce_add_plugin_wrapper_target(AUv3
      "${verbx_juce_plugin_client_path}" "${verbx_juce_modules_path}")
  endif()

  function(_juce_get_platform_plugin_kinds out)
    set(${out} Standalone AUv3 AU VST LV2 Unity VST3 AAX PARENT_SCOPE)
  endfunction()
endif()

add_subdirectory(../verbx_c verbx_c_core_build)

juce_add_plugin(VERBXPlugin
  COMPANY_NAME "Colby Leider"
  COMPANY_WEBSITE "https://github.com/TheColby/verbx"
  COMPANY_COPYRIGHT "Copyright 2026 Colby Leider"
  BUNDLE_ID "com.colbyleider.verbx"
  DESCRIPTION "High-resolution realtime reverberation"
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
    juce::juce_dsp
)

target_compile_definitions(VERBXPlugin
  PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
)

if(VERBX_BUILD_HOST_SMOKE_TEST)
  add_executable(VERBXEditorInteractionSmoke src/VerbXEditorInteractionSmoke.cpp)
  set_target_properties(VERBXEditorInteractionSmoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/VERBXEditorInteractionSmoke_artefacts"
  )
  target_include_directories(VERBXEditorInteractionSmoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../verbx_c/include
  )
  target_link_libraries(VERBXEditorInteractionSmoke PRIVATE
    VERBXPlugin
    juce::juce_audio_utils
  )
  target_compile_definitions(VERBXEditorInteractionSmoke PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
  )

  juce_add_console_app(VERBXPluginHostSmoke
    PRODUCT_NAME "VERBX Plug-in Host Smoke Test"
  )
  target_sources(VERBXPluginHostSmoke PRIVATE src/VerbXPluginHostSmoke.cpp)
  target_link_libraries(VERBXPluginHostSmoke PRIVATE juce::juce_audio_utils)
  target_compile_definitions(VERBXPluginHostSmoke PRIVATE
    JUCE_PLUGINHOST_AU=1
    JUCE_PLUGINHOST_VST3=1
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
  )
endif()

if(TARGET VERBXPlugin_AUv3 AND TARGET VERBXPlugin_Standalone
   AND NOT CMAKE_GENERATOR STREQUAL "Xcode")
  target_compile_options(VERBXPlugin_AUv3 PRIVATE -fapplication-extension)
  target_link_options(VERBXPlugin_AUv3 PRIVATE
    -fapplication-extension
    "LINKER:-e,_NSExtensionMain"
  )

  add_custom_command(TARGET VERBXPlugin_Standalone POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory
      "$<TARGET_BUNDLE_DIR:VERBXPlugin_Standalone>/Contents/PlugIns"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
      "$<TARGET_BUNDLE_DIR:VERBXPlugin_AUv3>"
      "$<TARGET_BUNDLE_DIR:VERBXPlugin_Standalone>/Contents/PlugIns/VERBX.appex"
    COMMENT "Embedding VERBX AUv3 extension in the standalone container"
    VERBATIM
  )
endif()
