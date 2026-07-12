#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>

#include <array>

class VerbXPluginProcessor;

class VerbXSpectrumAnalyzer final : public juce::Component, private juce::Timer {
public:
    explicit VerbXSpectrumAnalyzer(VerbXPluginProcessor& processor);
    ~VerbXSpectrumAnalyzer() override = default;

    void paint(juce::Graphics& graphics) override;

private:
    static constexpr int fftOrder = 13;
    static constexpr int fftSize = 1 << fftOrder;
    static constexpr int spectrumBins = fftSize / 2;
    static constexpr float floorDb = -96.0f;

    VerbXPluginProcessor& processor_;
    juce::dsp::FFT fft_{fftOrder};
    juce::dsp::WindowingFunction<float> window_{
        fftSize,
        juce::dsp::WindowingFunction<float>::hann,
        true
    };
    std::array<float, fftSize> history_{};
    std::array<float, fftSize * 2> fftData_{};
    std::array<float, spectrumBins> smoothedDb_{};
    std::array<float, spectrumBins> peakDb_{};
    std::array<float, 8192> drainBuffer_{};
    int historyWritePosition_ = 0;
    int historySampleCount_ = 0;

    void timerCallback() override;
    void appendSamples(const float* samples, int count) noexcept;
    void updateSpectrum();
    float frequencyToX(float frequency, juce::Rectangle<float> bounds) const;
    static float decibelsToY(float decibels, juce::Rectangle<float> bounds);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(VerbXSpectrumAnalyzer)
};

class VerbXPluginEditor final : public juce::AudioProcessorEditor {
public:
    explicit VerbXPluginEditor(VerbXPluginProcessor& processor);
    ~VerbXPluginEditor() override = default;

    void paint(juce::Graphics& graphics) override;
    void resized() override;

private:
    VerbXPluginProcessor& processor_;
    VerbXSpectrumAnalyzer spectrumAnalyzer_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(VerbXPluginEditor)
};
