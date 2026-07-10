#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

class VerbXPluginProcessor;

class VerbXPluginEditor final : public juce::AudioProcessorEditor {
public:
    explicit VerbXPluginEditor(VerbXPluginProcessor& processor);
    ~VerbXPluginEditor() override = default;

    void paint(juce::Graphics& graphics) override;
    void resized() override;

private:
    VerbXPluginProcessor& processor_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(VerbXPluginEditor)
};
