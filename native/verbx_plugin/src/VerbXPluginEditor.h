#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>

#include <array>
#include <memory>

class VerbXPluginProcessor;

class VerbXKnobSlider final : public juce::Slider {
public:
    void mouseDown(const juce::MouseEvent& event) override;

private:
    void setValueFromDialPosition(juce::Point<float> position);
};

class VerbXLookAndFeel final : public juce::LookAndFeel_V4 {
public:
    VerbXLookAndFeel();

    void drawRotarySlider(
        juce::Graphics& graphics,
        int x,
        int y,
        int width,
        int height,
        float sliderPosition,
        float rotaryStartAngle,
        float rotaryEndAngle,
        juce::Slider& slider
    ) override;
    void drawToggleButton(
        juce::Graphics& graphics,
        juce::ToggleButton& button,
        bool shouldDrawButtonAsHighlighted,
        bool shouldDrawButtonAsDown
    ) override;
    void drawComboBox(
        juce::Graphics& graphics,
        int width,
        int height,
        bool isButtonDown,
        int buttonX,
        int buttonY,
        int buttonWidth,
        int buttonHeight,
        juce::ComboBox& box
    ) override;
};

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

class VerbXPluginEditor final : public juce::AudioProcessorEditor, private juce::Timer {
public:
    explicit VerbXPluginEditor(VerbXPluginProcessor& processor);
    ~VerbXPluginEditor() override = default;

    void paint(juce::Graphics& graphics) override;
    void resized() override;

private:
    static constexpr int knobCount = 9;
    static constexpr int expertSelectGroupCount = 5;
    static constexpr int expertSelectsPerGroup = 4;
    static constexpr int expertSelectButtonCount =
        expertSelectGroupCount * expertSelectsPerGroup;
    enum class Page {
        perform,
        expert,
    };
    using SliderAttachment = juce::AudioProcessorValueTreeState::SliderAttachment;
    using ButtonAttachment = juce::AudioProcessorValueTreeState::ButtonAttachment;
    using ComboBoxAttachment = juce::AudioProcessorValueTreeState::ComboBoxAttachment;

    VerbXPluginProcessor& processor_;
    VerbXSpectrumAnalyzer spectrumAnalyzer_;
    VerbXLookAndFeel lookAndFeel_;
    juce::TooltipWindow tooltipWindow_;
    Page activePage_ = Page::perform;
    juce::TextButton performPageButton_{"PERFORM"};
    juce::TextButton expertPageButton_{"EXPERT"};
    std::array<VerbXKnobSlider, knobCount> knobs_{};
    std::array<juce::Label, knobCount> knobLabels_{};
    std::array<std::unique_ptr<SliderAttachment>, knobCount> knobAttachments_{};
    std::array<VerbXKnobSlider, knobCount> expertKnobs_{};
    std::array<juce::Label, knobCount> expertKnobLabels_{};
    std::array<std::unique_ptr<SliderAttachment>, knobCount> expertKnobAttachments_{};
    std::array<juce::Slider, knobCount> expertFaders_{};
    std::array<juce::Label, knobCount> expertFaderLabels_{};
    std::array<std::unique_ptr<SliderAttachment>, knobCount> expertFaderAttachments_{};
    std::array<juce::TextButton, expertSelectButtonCount> expertSelectButtons_{};
    juce::ToggleButton freezeButton_{"FREEZE"};
    juce::ToggleButton reverseButton_{"REVERSE"};
    juce::ComboBox qualityBox_;
    juce::Label qualityLabel_;
    juce::Label rt60Readout_;
    std::unique_ptr<ButtonAttachment> freezeAttachment_;
    std::unique_ptr<ButtonAttachment> reverseAttachment_;
    std::unique_ptr<ComboBoxAttachment> qualityAttachment_;

    void timerCallback() override;
    void configureControls();
    void configureExpertControls();
    void updatePageVisibility();
    void paintExpertPage(juce::Graphics& graphics);
    void setPlainParameter(const char* parameterId, float value);
    float plainParameter(const char* parameterId) const;
    void selectExpertMacro(int group, int option);
    void syncExpertMacroSelections();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(VerbXPluginEditor)
};
