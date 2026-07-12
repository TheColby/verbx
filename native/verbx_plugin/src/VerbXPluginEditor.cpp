#include "VerbXPluginEditor.h"
#include "VerbXPluginProcessor.h"

#include <algorithm>
#include <cmath>

namespace {

const auto analyzerMint = juce::Colour::fromRGB(140, 246, 210);
const auto analyzerGold = juce::Colour::fromRGB(213, 168, 75);
const auto analyzerInk = juce::Colour::fromRGB(5, 7, 9);

struct KnobDefinition {
    const char* parameterId;
    const char* label;
};

constexpr std::array<KnobDefinition, 9> knobDefinitions{{
    {"pre_delay_ms", "PRE-DELAY"},
    {"room_size", "ROOM SIZE"},
    {"rt60_coarse", "RT60 COARSE"},
    {"rt60_fine", "RT60 FINE"},
    {"damping", "DAMPING"},
    {"width", "WIDTH"},
    {"diffusion", "DIFFUSION"},
    {"wet", "WET"},
    {"dry", "DRY"},
}};

} // namespace

VerbXSpectrumAnalyzer::VerbXSpectrumAnalyzer(VerbXPluginProcessor& processor)
    : processor_(processor) {
    smoothedDb_.fill(floorDb);
    peakDb_.fill(floorDb);
    setInterceptsMouseClicks(false, false);
    startTimerHz(30);
}

void VerbXSpectrumAnalyzer::timerCallback() {
    for (int pass = 0; pass < 8; ++pass) {
        const auto received = processor_.popAnalyzerSamples(
            drainBuffer_.data(),
            static_cast<int>(drainBuffer_.size())
        );
        if (received <= 0) {
            break;
        }
        appendSamples(drainBuffer_.data(), received);
        if (received < static_cast<int>(drainBuffer_.size())) {
            break;
        }
    }

    if (historySampleCount_ >= fftSize) {
        updateSpectrum();
        repaint();
    }
}

void VerbXSpectrumAnalyzer::appendSamples(const float* samples, int count) noexcept {
    for (int sample = 0; sample < count; ++sample) {
        history_[static_cast<size_t>(historyWritePosition_)] = samples[sample];
        historyWritePosition_ = (historyWritePosition_ + 1) % fftSize;
        historySampleCount_ = juce::jmin(historySampleCount_ + 1, fftSize);
    }
}

void VerbXSpectrumAnalyzer::updateSpectrum() {
    std::fill(fftData_.begin(), fftData_.end(), 0.0f);
    for (int sample = 0; sample < fftSize; ++sample) {
        const auto source = (historyWritePosition_ + sample) % fftSize;
        fftData_[static_cast<size_t>(sample)] = history_[static_cast<size_t>(source)];
    }

    window_.multiplyWithWindowingTable(fftData_.data(), fftSize);
    fft_.performFrequencyOnlyForwardTransform(fftData_.data());
    const auto normalization = 2.0f / static_cast<float>(fftSize);
    for (int bin = 1; bin < spectrumBins; ++bin) {
        const auto magnitude = fftData_[static_cast<size_t>(bin)] * normalization;
        const auto current = juce::Decibels::gainToDecibels(magnitude, floorDb);
        auto& smoothed = smoothedDb_[static_cast<size_t>(bin)];
        smoothed = current >= smoothed ? current : juce::jmax(current, smoothed - 2.4f);
        auto& peak = peakDb_[static_cast<size_t>(bin)];
        peak = current >= peak ? current : juce::jmax(current, peak - 0.65f);
    }
}

float VerbXSpectrumAnalyzer::frequencyToX(
    float frequency,
    juce::Rectangle<float> bounds
) const {
    const auto nyquist = static_cast<float>(processor_.analyzerSampleRate() * 0.5);
    const auto maximum = juce::jmax(40.0f, juce::jmin(20000.0f, nyquist));
    const auto clamped = juce::jlimit(20.0f, maximum, frequency);
    const auto proportion = std::log10(clamped / 20.0f) / std::log10(maximum / 20.0f);
    return bounds.getX() + bounds.getWidth() * proportion;
}

float VerbXSpectrumAnalyzer::decibelsToY(
    float decibels,
    juce::Rectangle<float> bounds
) {
    const auto proportion = juce::jmap(juce::jlimit(floorDb, 0.0f, decibels), floorDb, 0.0f, 0.0f, 1.0f);
    return bounds.getBottom() - bounds.getHeight() * proportion;
}

void VerbXSpectrumAnalyzer::paint(juce::Graphics& graphics) {
    const auto panel = getLocalBounds().toFloat();
    const auto plot = panel.reduced(22.0f, 34.0f).withTrimmedTop(10.0f);

    graphics.setColour(analyzerInk.withAlpha(0.78f));
    graphics.fillRoundedRectangle(panel, 22.0f);
    graphics.setColour(analyzerMint.withAlpha(0.24f));
    graphics.drawRoundedRectangle(panel.reduced(0.5f), 22.0f, 1.0f);

    graphics.setFont(juce::FontOptions(11.0f, juce::Font::bold));
    graphics.setColour(analyzerMint.withAlpha(0.78f));
    graphics.drawText("POST  /  REALTIME FFT 8192  /  30 FPS", 22, 10, 300, 20, juce::Justification::centredLeft);
    graphics.setColour(juce::Colour::fromRGB(180, 197, 200).withAlpha(0.72f));
    graphics.drawText("-96 dBFS", getWidth() - 112, 10, 90, 20, juce::Justification::centredRight);

    constexpr std::array<float, 10> frequencies{
        20.0f, 50.0f, 100.0f, 200.0f, 500.0f,
        1000.0f, 2000.0f, 5000.0f, 10000.0f, 20000.0f
    };
    graphics.setFont(juce::FontOptions(10.0f));
    for (const auto frequency : frequencies) {
        if (frequency > processor_.analyzerSampleRate() * 0.5) {
            continue;
        }
        const auto x = frequencyToX(frequency, plot);
        graphics.setColour(juce::Colours::white.withAlpha(0.08f));
        graphics.drawVerticalLine(juce::roundToInt(x), plot.getY(), plot.getBottom());
        graphics.setColour(juce::Colour::fromRGB(180, 197, 200).withAlpha(0.66f));
        const auto label = frequency >= 1000.0f
            ? juce::String(frequency / 1000.0f, frequency < 10000.0f ? 1 : 0) + "k"
            : juce::String(juce::roundToInt(frequency));
        graphics.drawText(label, juce::roundToInt(x) - 18, juce::roundToInt(plot.getBottom()) + 5, 36, 16, juce::Justification::centred);
    }
    for (float db = floorDb; db <= 0.0f; db += 12.0f) {
        const auto y = decibelsToY(db, plot);
        graphics.setColour(juce::Colours::white.withAlpha(db == -48.0f ? 0.14f : 0.07f));
        graphics.drawHorizontalLine(juce::roundToInt(y), plot.getX(), plot.getRight());
    }

    const auto sampleRate = static_cast<float>(processor_.analyzerSampleRate());
    const auto firstBin = juce::jmax(1, juce::roundToInt(20.0f * fftSize / sampleRate));
    const auto lastFrequency = juce::jmin(20000.0f, sampleRate * 0.5f);
    const auto lastBin = juce::jmin(
        spectrumBins - 1,
        juce::roundToInt(lastFrequency * fftSize / sampleRate)
    );
    juce::Path spectrum;
    juce::Path peaks;
    for (int bin = firstBin; bin <= lastBin; ++bin) {
        const auto frequency = static_cast<float>(bin) * sampleRate / static_cast<float>(fftSize);
        const auto x = frequencyToX(frequency, plot);
        const auto y = decibelsToY(smoothedDb_[static_cast<size_t>(bin)], plot);
        const auto peakY = decibelsToY(peakDb_[static_cast<size_t>(bin)], plot);
        if (bin == firstBin) {
            spectrum.startNewSubPath(x, y);
            peaks.startNewSubPath(x, peakY);
        } else {
            spectrum.lineTo(x, y);
            peaks.lineTo(x, peakY);
        }
    }

    auto fill = spectrum;
    fill.lineTo(plot.getRight(), plot.getBottom());
    fill.lineTo(plot.getX(), plot.getBottom());
    fill.closeSubPath();
    juce::ColourGradient gradient(
        analyzerMint.withAlpha(0.42f), plot.getCentreX(), plot.getY(),
        analyzerMint.withAlpha(0.015f), plot.getCentreX(), plot.getBottom(), false
    );
    graphics.setGradientFill(gradient);
    graphics.fillPath(fill);
    graphics.setColour(analyzerMint.withAlpha(0.95f));
    graphics.strokePath(spectrum, juce::PathStrokeType(1.8f));
    graphics.setColour(analyzerGold.withAlpha(0.58f));
    graphics.strokePath(peaks, juce::PathStrokeType(0.8f));
}

VerbXPluginEditor::VerbXPluginEditor(VerbXPluginProcessor& processor)
    : AudioProcessorEditor(&processor), processor_(processor), spectrumAnalyzer_(processor) {
    addAndMakeVisible(spectrumAnalyzer_);
    configureControls();
    setResizable(true, true);
    setResizeLimits(960, 600, 2560, 1600);
    setSize(1560, 920);
    startTimerHz(15);
}

void VerbXPluginEditor::configureControls() {
    auto& state = processor_.state();
    for (int index = 0; index < knobCount; ++index) {
        const auto& definition = knobDefinitions[static_cast<size_t>(index)];
        auto& knob = knobs_[static_cast<size_t>(index)];
        auto& label = knobLabels_[static_cast<size_t>(index)];
        knob.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
        knob.setTextBoxStyle(juce::Slider::TextBoxBelow, true, 76, 20);
        knob.setColour(juce::Slider::rotarySliderFillColourId, analyzerMint);
        knob.setColour(juce::Slider::rotarySliderOutlineColourId, juce::Colours::white.withAlpha(0.12f));
        knob.setColour(juce::Slider::thumbColourId, analyzerGold);
        knob.setColour(juce::Slider::textBoxTextColourId, juce::Colour::fromRGB(228, 240, 236));
        knob.setColour(juce::Slider::textBoxBackgroundColourId, analyzerInk.withAlpha(0.72f));
        knob.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
        addAndMakeVisible(knob);

        label.setText(definition.label, juce::dontSendNotification);
        label.setJustificationType(juce::Justification::centred);
        label.setFont(juce::FontOptions(10.0f, juce::Font::bold));
        label.setColour(juce::Label::textColourId, juce::Colour::fromRGB(180, 197, 200));
        addAndMakeVisible(label);
        knobAttachments_[static_cast<size_t>(index)] = std::make_unique<SliderAttachment>(
            state,
            definition.parameterId,
            knob
        );
        if (index == 0) {
            knob.textFromValueFunction = [](double value) {
                return juce::String(value, 1) + " ms";
            };
        } else if (index == 2) {
            knob.textFromValueFunction = [](double value) {
                const auto seconds = verbx_plugin_map_rt60_seconds(value, 0.0);
                const auto precision = seconds < 1.0 ? 3 : (seconds < 10.0 ? 2 : 1);
                return juce::String(seconds, precision) + " s";
            };
        } else if (index == 3) {
            knob.textFromValueFunction = [](double value) {
                const auto percent = juce::roundToInt(value * 20.0);
                return juce::String(percent >= 0 ? "+" : "") + juce::String(percent) + "%";
            };
        } else {
            knob.textFromValueFunction = [](double value) {
                return juce::String(juce::roundToInt(value * 100.0)) + "%";
            };
        }
        knob.updateText();
    }

    for (auto* button : {&freezeButton_, &reverseButton_}) {
        button->setColour(juce::ToggleButton::textColourId, juce::Colour::fromRGB(228, 240, 236));
        button->setColour(juce::ToggleButton::tickColourId, analyzerMint);
        button->setColour(juce::ToggleButton::tickDisabledColourId, juce::Colours::white.withAlpha(0.18f));
        addAndMakeVisible(*button);
    }
    freezeAttachment_ = std::make_unique<ButtonAttachment>(state, "freeze", freezeButton_);
    reverseAttachment_ = std::make_unique<ButtonAttachment>(state, "reverse", reverseButton_);

    qualityBox_.addItemList({"Host", "2x", "4x", "Target 192 kHz"}, 1);
    qualityBox_.setColour(juce::ComboBox::backgroundColourId, analyzerInk.brighter(0.12f));
    qualityBox_.setColour(juce::ComboBox::textColourId, juce::Colour::fromRGB(228, 240, 236));
    qualityBox_.setColour(juce::ComboBox::outlineColourId, analyzerMint.withAlpha(0.35f));
    addAndMakeVisible(qualityBox_);
    qualityAttachment_ = std::make_unique<ComboBoxAttachment>(state, "quality_mode", qualityBox_);

    qualityLabel_.setText("QUALITY", juce::dontSendNotification);
    qualityLabel_.setJustificationType(juce::Justification::centredLeft);
    qualityLabel_.setFont(juce::FontOptions(10.0f, juce::Font::bold));
    qualityLabel_.setColour(juce::Label::textColourId, juce::Colour::fromRGB(180, 197, 200));
    addAndMakeVisible(qualityLabel_);

    rt60Readout_.setJustificationType(juce::Justification::centredRight);
    rt60Readout_.setFont(juce::FontOptions(15.0f, juce::Font::bold));
    rt60Readout_.setColour(juce::Label::textColourId, analyzerGold);
    addAndMakeVisible(rt60Readout_);
    timerCallback();
}

void VerbXPluginEditor::timerCallback() {
    const auto seconds = processor_.effectiveRt60Seconds();
    const auto precision = seconds < 1.0 ? 3 : (seconds < 10.0 ? 2 : 1);
    rt60Readout_.setText(
        "EFFECTIVE RT60  " + juce::String(seconds, precision) + " s",
        juce::dontSendNotification
    );
}

void VerbXPluginEditor::paint(juce::Graphics& graphics) {
    const auto bounds = getLocalBounds().toFloat();
    graphics.fillAll(juce::Colour::fromRGB(5, 7, 9));

    graphics.setColour(juce::Colour::fromRGB(140, 246, 210));
    graphics.setFont(juce::FontOptions(34.0f, juce::Font::bold));
    graphics.drawText("VERBX", 28, 24, 220, 48, juce::Justification::centredLeft);

    graphics.setColour(juce::Colour::fromRGB(180, 197, 200));
    graphics.setFont(juce::FontOptions(15.0f));
    graphics.drawText(
        "Spatial Decay Theater | Target 192 kHz | RT60 0.01s to 360s",
        28,
        76,
        static_cast<int>(bounds.getWidth()) - 56,
        28,
        juce::Justification::centredLeft
    );

    const auto console = bounds.reduced(28.0f, 130.0f);
    graphics.setColour(juce::Colour::fromRGBA(140, 246, 210, 36));
    graphics.fillRoundedRectangle(console, 24.0f);
    graphics.setColour(juce::Colour::fromRGBA(232, 240, 247, 48));
    graphics.drawRoundedRectangle(console, 24.0f, 1.0f);

    graphics.setColour(juce::Colour::fromRGB(180, 197, 200).withAlpha(0.62f));
    graphics.setFont(juce::FontOptions(11.0f, juce::Font::bold));
    graphics.drawText("LIVE SPECTRAL FIELD", 52, 116, 240, 18, juce::Justification::centredLeft);

    const auto dock = getLocalBounds().toFloat().reduced(28.0f).removeFromBottom(190.0f);
    graphics.setColour(juce::Colour::fromRGB(10, 17, 19).withAlpha(0.96f));
    graphics.fillRoundedRectangle(dock, 18.0f);
    graphics.setColour(analyzerMint.withAlpha(0.16f));
    graphics.drawRoundedRectangle(dock.reduced(0.5f), 18.0f, 1.0f);
}

void VerbXPluginEditor::resized() {
    auto content = getLocalBounds().reduced(28);
    content.removeFromTop(112);
    auto controls = content.removeFromBottom(190).reduced(12, 8);
    spectrumAnalyzer_.setBounds(content.reduced(16, 8));

    auto modes = controls.removeFromRight(190).reduced(8, 4);
    const auto knobWidth = controls.getWidth() / knobCount;
    for (int index = 0; index < knobCount; ++index) {
        auto cell = index == knobCount - 1
            ? controls
            : controls.removeFromLeft(knobWidth);
        auto& label = knobLabels_[static_cast<size_t>(index)];
        label.setBounds(cell.removeFromTop(20));
        knobs_[static_cast<size_t>(index)].setBounds(cell.reduced(2));
    }

    qualityLabel_.setBounds(modes.removeFromTop(20));
    qualityBox_.setBounds(modes.removeFromTop(30));
    modes.removeFromTop(8);
    freezeButton_.setBounds(modes.removeFromTop(30));
    reverseButton_.setBounds(modes.removeFromTop(30));
    rt60Readout_.setBounds(getWidth() - 330, 70, 300, 28);
}
