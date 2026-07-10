#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

extern "C" {
#include "verbx_c/plugin_realtime.h"
}

class VerbXPluginProcessor final : public juce::AudioProcessor {
public:
    VerbXPluginProcessor();
    ~VerbXPluginProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState& state();

private:
    juce::AudioProcessorValueTreeState parameters_;
    verbx_plugin_realtime_context realtimeContext_{};

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    verbx_plugin_realtime_params currentRealtimeParams() const;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(VerbXPluginProcessor)
};
