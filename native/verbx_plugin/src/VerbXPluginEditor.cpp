#include "VerbXPluginEditor.h"
#include "VerbXPluginProcessor.h"

#include <algorithm>
#include <cmath>

namespace {

const auto analyzerMint = juce::Colour::fromRGB(140, 246, 210);
const auto analyzerGold = juce::Colour::fromRGB(213, 168, 75);
const auto analyzerInk = juce::Colour::fromRGB(5, 7, 9);
const auto consolePanel = juce::Colour::fromRGB(12, 18, 22);
const auto consoleLine = juce::Colour::fromRGB(71, 88, 94);
const auto consoleText = juce::Colour::fromRGB(220, 228, 228);
const auto consoleMuted = juce::Colour::fromRGB(132, 150, 154);
const auto consoleCoral = juce::Colour::fromRGB(240, 112, 130);

constexpr float designWidth = 1920.0f;
constexpr float designHeight = 1080.0f;

juce::Font consoleFont(float size, int style = juce::Font::plain) {
    auto font = juce::Font(juce::FontOptions(size, style));
    font.setTypefaceName("Avenir Next Condensed");
    return font;
}

juce::Font dataFont(float size, int style = juce::Font::plain) {
    auto font = juce::Font(juce::FontOptions(size, style));
    font.setTypefaceName("Menlo");
    return font;
}

void drawPanel(
    juce::Graphics& graphics,
    juce::Rectangle<float> bounds,
    const juce::String& title,
    const juce::String& detail = {}
) {
    graphics.setColour(consolePanel.withAlpha(0.92f));
    graphics.fillRoundedRectangle(bounds, 16.0f);
    graphics.setColour(consoleLine.withAlpha(0.42f));
    graphics.drawRoundedRectangle(bounds.reduced(0.5f), 16.0f, 1.0f);
    graphics.drawHorizontalLine(
        juce::roundToInt(bounds.getY() + 42.0f),
        bounds.getX(),
        bounds.getRight()
    );

    graphics.setFont(dataFont(11.0f, juce::Font::bold));
    graphics.setColour(consoleText.withAlpha(0.92f));
    graphics.drawText(title, bounds.getX() + 16.0f, bounds.getY() + 10.0f,
                      bounds.getWidth() - 32.0f, 22.0f, juce::Justification::centredLeft);
    if (detail.isNotEmpty()) {
        graphics.setColour(consoleMuted);
        graphics.drawText(detail, bounds.getX() + 16.0f, bounds.getY() + 10.0f,
                          bounds.getWidth() - 32.0f, 22.0f, juce::Justification::centredRight);
    }
}

void drawDataCard(
    juce::Graphics& graphics,
    juce::Rectangle<float> bounds,
    const juce::String& label,
    const juce::String& value,
    float level
) {
    graphics.setColour(juce::Colour::fromRGB(15, 21, 24));
    graphics.fillRoundedRectangle(bounds, 12.0f);
    graphics.setColour(consoleLine.withAlpha(0.34f));
    graphics.drawRoundedRectangle(bounds.reduced(0.5f), 12.0f, 1.0f);
    graphics.setColour(consoleMuted);
    graphics.setFont(dataFont(9.5f, juce::Font::bold));
    graphics.drawText(label, bounds.getX() + 12.0f, bounds.getY() + 9.0f,
                      bounds.getWidth() - 24.0f, 16.0f, juce::Justification::centredLeft);
    graphics.setColour(consoleText);
    graphics.setFont(consoleFont(17.0f, juce::Font::bold));
    graphics.drawText(value, bounds.getX() + 12.0f, bounds.getY() + 30.0f,
                      bounds.getWidth() - 24.0f, 23.0f, juce::Justification::centredLeft);
    const auto bar = juce::Rectangle<float>(bounds.getX() + 12.0f, bounds.getBottom() - 18.0f,
                                            bounds.getWidth() - 24.0f, 5.0f);
    graphics.setColour(juce::Colours::white.withAlpha(0.08f));
    graphics.fillRoundedRectangle(bar, 2.5f);
    graphics.setColour(analyzerMint.withAlpha(0.9f));
    graphics.fillRoundedRectangle(bar.withWidth(bar.getWidth() * juce::jlimit(0.0f, 1.0f, level)), 2.5f);
}

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

VerbXLookAndFeel::VerbXLookAndFeel() {
    setColour(juce::Label::textColourId, consoleText);
    setColour(juce::ComboBox::textColourId, consoleText);
    setColour(juce::PopupMenu::backgroundColourId, consolePanel);
    setColour(juce::PopupMenu::textColourId, consoleText);
    setColour(juce::PopupMenu::highlightedBackgroundColourId, analyzerMint.withAlpha(0.18f));
}

void VerbXLookAndFeel::drawRotarySlider(
    juce::Graphics& graphics,
    int x,
    int y,
    int width,
    int height,
    float sliderPosition,
    float rotaryStartAngle,
    float rotaryEndAngle,
    juce::Slider& slider
) {
    juce::ignoreUnused(slider);
    const auto diameter = static_cast<float>(juce::jmin(width, height)) - 10.0f;
    const auto bounds = juce::Rectangle<float>(
        static_cast<float>(x) + (static_cast<float>(width) - diameter) * 0.5f,
        static_cast<float>(y) + (static_cast<float>(height) - diameter) * 0.5f,
        diameter,
        diameter
    );
    const auto angle = rotaryStartAngle + sliderPosition * (rotaryEndAngle - rotaryStartAngle);

    graphics.setColour(juce::Colours::black.withAlpha(0.42f));
    graphics.fillEllipse(bounds.translated(0.0f, 4.0f));
    graphics.setColour(juce::Colour::fromRGB(31, 39, 43));
    graphics.fillEllipse(bounds);
    graphics.setColour(consoleLine.withAlpha(0.58f));
    graphics.drawEllipse(bounds.reduced(0.5f), 1.0f);

    juce::Path valueWedge;
    valueWedge.addPieSegment(bounds.reduced(5.0f), rotaryStartAngle, angle, 0.56f);
    graphics.setColour(analyzerMint.withAlpha(0.96f));
    graphics.fillPath(valueWedge);

    juce::Path pointer;
    const auto pointerLength = diameter * 0.38f;
    pointer.addRoundedRectangle(-2.0f, -pointerLength, 4.0f, pointerLength, 2.0f);
    graphics.setColour(consoleText.withAlpha(0.9f));
    graphics.fillPath(pointer, juce::AffineTransform::rotation(angle).translated(bounds.getCentreX(), bounds.getCentreY()));
    graphics.setColour(analyzerGold);
    graphics.fillEllipse(bounds.getCentreX() - 4.5f, bounds.getCentreY() - 4.5f, 9.0f, 9.0f);
}

void VerbXLookAndFeel::drawToggleButton(
    juce::Graphics& graphics,
    juce::ToggleButton& button,
    bool shouldDrawButtonAsHighlighted,
    bool shouldDrawButtonAsDown
) {
    auto bounds = button.getLocalBounds().toFloat().reduced(1.0f);
    const auto active = button.getToggleState();
    graphics.setColour(active ? analyzerMint.withAlpha(0.18f) : consolePanel);
    graphics.fillRoundedRectangle(bounds, bounds.getHeight() * 0.5f);
    graphics.setColour(active ? analyzerMint.withAlpha(0.75f) : consoleLine.withAlpha(0.55f));
    graphics.drawRoundedRectangle(bounds, bounds.getHeight() * 0.5f, shouldDrawButtonAsDown ? 2.0f : 1.0f);
    const auto light = bounds.removeFromLeft(bounds.getHeight()).reduced(7.0f);
    graphics.setColour(active ? analyzerMint : consoleMuted.withAlpha(0.42f));
    graphics.fillEllipse(light);
    graphics.setColour((shouldDrawButtonAsHighlighted ? juce::Colours::white : consoleText).withAlpha(0.92f));
    graphics.setFont(dataFont(10.0f, juce::Font::bold));
    graphics.drawText(button.getButtonText(), button.getLocalBounds().reduced(34, 0), juce::Justification::centredLeft);
}

void VerbXLookAndFeel::drawComboBox(
    juce::Graphics& graphics,
    int width,
    int height,
    bool isButtonDown,
    int buttonX,
    int buttonY,
    int buttonWidth,
    int buttonHeight,
    juce::ComboBox& box
) {
    juce::ignoreUnused(buttonX, buttonY, buttonWidth, buttonHeight, box);
    const auto bounds = juce::Rectangle<float>(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height)).reduced(0.5f);
    graphics.setColour(consolePanel.brighter(isButtonDown ? 0.12f : 0.05f));
    graphics.fillRoundedRectangle(bounds, 10.0f);
    graphics.setColour(analyzerMint.withAlpha(0.36f));
    graphics.drawRoundedRectangle(bounds, 10.0f, 1.0f);
    juce::Path arrow;
    arrow.addTriangle(static_cast<float>(width - 20), static_cast<float>(height) * 0.42f,
                      static_cast<float>(width - 10), static_cast<float>(height) * 0.42f,
                      static_cast<float>(width - 15), static_cast<float>(height) * 0.62f);
    graphics.setColour(analyzerMint);
    graphics.fillPath(arrow);
}

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
    setResizeLimits(1184, 666, 2560, 1440);
    if (auto* constrainer = getConstrainer()) {
        constrainer->setFixedAspectRatio(16.0 / 9.0);
    }
    setSize(1728, 972);
    startTimerHz(15);
}

void VerbXPluginEditor::configureControls() {
    auto& state = processor_.state();
    for (int index = 0; index < knobCount; ++index) {
        const auto& definition = knobDefinitions[static_cast<size_t>(index)];
        auto& knob = knobs_[static_cast<size_t>(index)];
        auto& label = knobLabels_[static_cast<size_t>(index)];
        knob.setLookAndFeel(&lookAndFeel_);
        knob.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
        knob.setTextBoxStyle(juce::Slider::TextBoxBelow, true, 80, 18);
        knob.setColour(juce::Slider::rotarySliderFillColourId, analyzerMint);
        knob.setColour(juce::Slider::rotarySliderOutlineColourId, juce::Colours::white.withAlpha(0.12f));
        knob.setColour(juce::Slider::thumbColourId, analyzerGold);
        knob.setColour(juce::Slider::textBoxTextColourId, juce::Colour::fromRGB(228, 240, 236));
        knob.setColour(juce::Slider::textBoxBackgroundColourId, analyzerInk.withAlpha(0.72f));
        knob.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
        addAndMakeVisible(knob);

        label.setText(definition.label, juce::dontSendNotification);
        label.setJustificationType(juce::Justification::centred);
        label.setFont(dataFont(9.5f, juce::Font::bold));
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
        button->setLookAndFeel(&lookAndFeel_);
        button->setColour(juce::ToggleButton::textColourId, juce::Colour::fromRGB(228, 240, 236));
        button->setColour(juce::ToggleButton::tickColourId, analyzerMint);
        button->setColour(juce::ToggleButton::tickDisabledColourId, juce::Colours::white.withAlpha(0.18f));
        addAndMakeVisible(*button);
    }
    freezeAttachment_ = std::make_unique<ButtonAttachment>(state, "freeze", freezeButton_);
    reverseAttachment_ = std::make_unique<ButtonAttachment>(state, "reverse", reverseButton_);

    qualityBox_.addItemList({"Host", "2x", "4x", "Target 192 kHz"}, 1);
    qualityBox_.setLookAndFeel(&lookAndFeel_);
    qualityBox_.setColour(juce::ComboBox::backgroundColourId, analyzerInk.brighter(0.12f));
    qualityBox_.setColour(juce::ComboBox::textColourId, juce::Colour::fromRGB(228, 240, 236));
    qualityBox_.setColour(juce::ComboBox::outlineColourId, analyzerMint.withAlpha(0.35f));
    addAndMakeVisible(qualityBox_);
    qualityAttachment_ = std::make_unique<ComboBoxAttachment>(state, "quality_mode", qualityBox_);

    qualityLabel_.setText("QUALITY", juce::dontSendNotification);
    qualityLabel_.setJustificationType(juce::Justification::centredLeft);
    qualityLabel_.setFont(dataFont(9.5f, juce::Font::bold));
    qualityLabel_.setColour(juce::Label::textColourId, juce::Colour::fromRGB(180, 197, 200));
    addAndMakeVisible(qualityLabel_);

    rt60Readout_.setJustificationType(juce::Justification::centredRight);
    rt60Readout_.setFont(dataFont(14.0f, juce::Font::bold));
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
    graphics.fillAll(analyzerInk);
    const auto scale = juce::jmin(
        static_cast<float>(getWidth()) / designWidth,
        static_cast<float>(getHeight()) / designHeight
    );
    const auto offsetX = (static_cast<float>(getWidth()) - designWidth * scale) * 0.5f;
    const auto offsetY = (static_cast<float>(getHeight()) - designHeight * scale) * 0.5f;
    juce::Graphics::ScopedSaveState state(graphics);
    graphics.addTransform(juce::AffineTransform(scale, 0.0f, offsetX, 0.0f, scale, offsetY));

    graphics.setColour(juce::Colour::fromRGB(8, 14, 18));
    graphics.fillRect(0.0f, 0.0f, designWidth, designHeight);
    graphics.setColour(analyzerMint.withAlpha(0.025f));
    for (int x = 0; x < static_cast<int>(designWidth); x += 64) {
        graphics.drawVerticalLine(x, 0.0f, designHeight);
    }
    for (int y = 0; y < static_cast<int>(designHeight); y += 64) {
        graphics.drawHorizontalLine(y, 0.0f, designWidth);
    }
    graphics.setColour(analyzerMint.withAlpha(0.035f));
    graphics.fillEllipse(500.0f, -420.0f, 1100.0f, 840.0f);

    const auto topBar = juce::Rectangle<float>(40.0f, 20.0f, 1840.0f, 72.0f);
    graphics.setColour(juce::Colour::fromRGB(22, 29, 34).withAlpha(0.94f));
    graphics.fillRoundedRectangle(topBar, 18.0f);
    graphics.setColour(consoleLine.withAlpha(0.42f));
    graphics.drawRoundedRectangle(topBar.reduced(0.5f), 18.0f, 1.0f);
    graphics.setColour(consoleText);
    graphics.setFont(consoleFont(29.0f, juce::Font::bold));
    graphics.drawText("V E R B X", 60, 30, 150, 38, juce::Justification::centredLeft);
    graphics.setColour(analyzerMint);
    graphics.setFont(dataFont(9.0f, juce::Font::bold));
    graphics.drawText("SPATIAL\nENGINE", 188, 38, 80, 34, juce::Justification::centredLeft);

    const auto preset = juce::Rectangle<float>(284.0f, 35.0f, 1220.0f, 42.0f);
    graphics.setColour(juce::Colours::black.withAlpha(0.18f));
    graphics.fillRoundedRectangle(preset, 20.0f);
    graphics.setColour(consoleLine.withAlpha(0.34f));
    graphics.drawRoundedRectangle(preset, 20.0f, 1.0f);
    graphics.setColour(consoleMuted);
    graphics.setFont(consoleFont(13.0f));
    graphics.drawText("Preset", 304, 44, 52, 22, juce::Justification::centredLeft);
    graphics.setColour(consoleText);
    graphics.setFont(consoleFont(14.0f, juce::Font::bold));
    graphics.drawText("DXF Hall  ·  Slow Bloom  ·  7.2.4", 360, 44, 470, 22, juce::Justification::centredLeft);
    graphics.setColour(consoleMuted);
    graphics.drawText("Browse", 1410, 44, 70, 22, juce::Justification::centredRight);

    const std::array<juce::String, 3> topModes{"ALGO", "CONV", "GEO"};
    for (size_t index = 0; index < topModes.size(); ++index) {
        const auto mode = juce::Rectangle<float>(1538.0f + static_cast<float>(index) * 56.0f, 37.0f, 52.0f, 38.0f);
        graphics.setColour(index == 0 ? analyzerMint.withAlpha(0.92f) : consolePanel);
        graphics.fillRoundedRectangle(mode, 18.0f);
        graphics.setColour(index == 0 ? analyzerInk : consoleMuted);
        graphics.setFont(dataFont(9.0f, juce::Font::bold));
        graphics.drawText(topModes[index], mode, juce::Justification::centred);
    }
    graphics.setColour(consolePanel);
    graphics.fillRoundedRectangle(1720.0f, 37.0f, 132.0f, 38.0f, 18.0f);
    graphics.setColour(analyzerMint);
    graphics.fillEllipse(1735.0f, 52.0f, 8.0f, 8.0f);
    graphics.setFont(dataFont(10.0f, juce::Font::bold));
    graphics.drawText("LIVE", 1748, 44, 80, 22, juce::Justification::centredLeft);

    const auto loudness = juce::Rectangle<float>(40.0f, 108.0f, 260.0f, 510.0f);
    const auto theater = juce::Rectangle<float>(318.0f, 108.0f, 930.0f, 510.0f);
    const auto imagePanel = juce::Rectangle<float>(1262.0f, 108.0f, 306.0f, 510.0f);
    const auto spacePanel = juce::Rectangle<float>(1582.0f, 108.0f, 298.0f, 510.0f);
    drawPanel(graphics, loudness, "LOUDNESS", "BS.1770");
    drawPanel(graphics, theater, "SPATIAL DECAY THEATER", "GEOMETRY IS THE HERO");
    drawPanel(graphics, imagePanel, "IMAGE", "7.2.4");
    drawPanel(graphics, spacePanel, "SPACE", "RAY MODEL");

    drawDataCard(graphics, {54.0f, 160.0f, 110.0f, 70.0f}, "INT", "-14.1", 0.72f);
    drawDataCard(graphics, {176.0f, 160.0f, 110.0f, 70.0f}, "TP", "-1.0", 0.9f);
    constexpr std::array<float, 8> meterValues{0.64f, 0.72f, 0.56f, 0.78f, 0.59f, 0.69f, 0.42f, 0.34f};
    for (size_t index = 0; index < meterValues.size(); ++index) {
        const auto x = 54.0f + static_cast<float>(index) * 29.0f;
        const auto track = juce::Rectangle<float>(x, 245.0f, 21.0f, 240.0f);
        graphics.setColour(juce::Colours::black.withAlpha(0.34f));
        graphics.fillRoundedRectangle(track, 9.0f);
        const auto fill = track.withTop(track.getBottom() - track.getHeight() * meterValues[index]);
        juce::ColourGradient meterGradient(analyzerMint, fill.getCentreX(), fill.getBottom(),
                                           consoleCoral, fill.getCentreX(), fill.getY(), false);
        meterGradient.addColour(0.55, juce::Colour::fromRGB(244, 212, 104));
        graphics.setGradientFill(meterGradient);
        graphics.fillRoundedRectangle(fill, 8.0f);
    }
    const std::array<std::pair<juce::String, bool>, 3> loudnessRows{{
        {"True Peak Limiter", true}, {"Duck Reverb", true}, {"Safety Gain     -2.0 dB", false}
    }};
    for (size_t index = 0; index < loudnessRows.size(); ++index) {
        const auto y = 510.0f + static_cast<float>(index) * 31.0f;
        graphics.setColour(consoleMuted);
        graphics.setFont(consoleFont(12.0f));
        graphics.drawText(loudnessRows[index].first, 54, juce::roundToInt(y), 180, 22, juce::Justification::centredLeft);
        if (loudnessRows[index].second) {
            graphics.setColour(analyzerMint.withAlpha(0.16f));
            graphics.fillRoundedRectangle(245.0f, y + 2.0f, 42.0f, 20.0f, 10.0f);
            graphics.setColour(analyzerMint);
            graphics.fillEllipse(268.0f, y + 5.0f, 14.0f, 14.0f);
        }
    }

    const auto shell = juce::Rectangle<float>(332.0f, 160.0f, 902.0f, 292.0f);
    graphics.setColour(juce::Colour::fromRGB(6, 12, 16));
    graphics.fillRoundedRectangle(shell, 14.0f);
    graphics.setColour(analyzerMint.withAlpha(0.055f));
    for (int x = 350; x < 1230; x += 44) graphics.drawVerticalLine(x, shell.getY(), shell.getBottom());
    for (int y = 182; y < 450; y += 34) graphics.drawHorizontalLine(y, shell.getX(), shell.getRight());
    graphics.setColour(consoleMuted);
    graphics.setFont(dataFont(10.0f));
    graphics.drawText("IMPORTED ACOUSTIC SHELL", 352, 178, 250, 18, juce::Justification::centredLeft);
    graphics.setColour(consoleText);
    graphics.setFont(consoleFont(19.0f, juce::Font::bold));
    graphics.drawText("GRAND ATRIUM DXF", 352, 198, 300, 26, juce::Justification::centredLeft);
    juce::Path room;
    room.startNewSubPath(392.0f, 385.0f);
    room.lineTo(468.0f, 267.0f);
    room.lineTo(1028.0f, 232.0f);
    room.lineTo(1148.0f, 326.0f);
    room.lineTo(392.0f, 385.0f);
    graphics.setColour(consoleLine.withAlpha(0.82f));
    graphics.strokePath(room, juce::PathStrokeType(1.6f));
    const juce::Point<float> source(782.0f, 314.0f);
    const std::array<juce::Point<float>, 6> rayEnds{{
        {548.0f, 166.0f}, {824.0f, 166.0f}, {1010.0f, 165.0f},
        {1166.0f, 438.0f}, {560.0f, 448.0f}, {1008.0f, 442.0f}
    }};
    for (size_t index = 0; index < rayEnds.size(); ++index) {
        graphics.setColour((index % 3 == 0 ? consoleCoral : (index % 3 == 1 ? analyzerMint : analyzerGold)).withAlpha(0.32f));
        graphics.drawLine({source, rayEnds[index]}, 1.4f);
    }
    graphics.setColour(analyzerMint);
    graphics.fillEllipse(source.x - 12.0f, source.y - 12.0f, 24.0f, 24.0f);
    graphics.setColour(analyzerGold);
    graphics.drawEllipse(966.0f, 275.0f, 20.0f, 20.0f, 2.0f);

    for (int index = 0; index < knobCount; ++index) {
        const auto cardX = 332.0f + static_cast<float>(index) * 99.0f;
        graphics.setColour(juce::Colour::fromRGB(17, 23, 27));
        graphics.fillRoundedRectangle(cardX, 468.0f, 92.0f, 136.0f, 12.0f);
        graphics.setColour(consoleLine.withAlpha(0.32f));
        graphics.drawRoundedRectangle(cardX + 0.5f, 468.5f, 91.0f, 135.0f, 12.0f, 1.0f);
    }

    const auto imagePlot = juce::Rectangle<float>(1276.0f, 160.0f, 278.0f, 338.0f);
    graphics.setColour(juce::Colour::fromRGB(16, 22, 26));
    graphics.fillRoundedRectangle(imagePlot, 14.0f);
    graphics.setColour(consoleLine.withAlpha(0.3f));
    graphics.drawRoundedRectangle(imagePlot, 14.0f, 1.0f);
    const auto imageCentre = imagePlot.getCentre();
    for (float radius : {55.0f, 92.0f, 132.0f}) {
        graphics.setColour(consoleLine.withAlpha(0.22f));
        graphics.drawEllipse(imageCentre.x - radius, imageCentre.y - radius,
                             radius * 2.0f, radius * 2.0f, 1.0f);
    }
    for (int offset = -84; offset <= 84; offset += 42) {
        graphics.drawVerticalLine(juce::roundToInt(imageCentre.x + static_cast<float>(offset)),
                                  imagePlot.getY() + 38.0f, imagePlot.getBottom() - 38.0f);
    }
    juce::Path orbit;
    orbit.addEllipse(imageCentre.x - 92.0f, imageCentre.y - 56.0f, 184.0f, 112.0f);
    graphics.setColour(analyzerMint.withAlpha(0.86f));
    graphics.strokePath(orbit, juce::PathStrokeType(2.0f));
    drawDataCard(graphics, {1276.0f, 512.0f, 132.0f, 90.0f}, "CORR", "+0.78", 0.78f);
    drawDataCard(graphics, {1420.0f, 512.0f, 134.0f, 90.0f}, "ORDER", "30A", 0.62f);

    const auto rayPlot = juce::Rectangle<float>(1596.0f, 160.0f, 270.0f, 294.0f);
    graphics.setColour(juce::Colour::fromRGB(16, 22, 26));
    graphics.fillRoundedRectangle(rayPlot, 14.0f);
    graphics.setColour(consoleLine.withAlpha(0.35f));
    graphics.drawRoundedRectangle(rayPlot, 14.0f, 1.0f);
    juce::Path triangle;
    triangle.startNewSubPath(1620.0f, 426.0f);
    triangle.lineTo(1654.0f, 244.0f);
    triangle.lineTo(1850.0f, 386.0f);
    triangle.closeSubPath();
    graphics.setColour(consoleLine.withAlpha(0.8f));
    graphics.strokePath(triangle, juce::PathStrokeType(2.0f));
    graphics.setColour(analyzerMint.withAlpha(0.38f));
    graphics.drawLine(1598.0f, 230.0f, 1864.0f, 385.0f, 3.0f);
    graphics.setColour(analyzerGold.withAlpha(0.3f));
    graphics.drawLine(1610.0f, 410.0f, 1865.0f, 186.0f, 6.0f);
    const std::array<std::pair<juce::String, juce::String>, 3> spaceRows{{
        {"Material", "Stone / Glass"}, {"Volume", "18,420 m3"}, {"Rays", "64k"}
    }};
    for (size_t index = 0; index < spaceRows.size(); ++index) {
        const auto y = 466.0f + static_cast<float>(index) * 34.0f;
        graphics.setColour(juce::Colour::fromRGB(18, 25, 29));
        graphics.fillRoundedRectangle(1596.0f, y, 270.0f, 28.0f, 9.0f);
        graphics.setFont(consoleFont(11.0f));
        graphics.setColour(consoleMuted);
        graphics.drawText(spaceRows[index].first, 1608, juce::roundToInt(y + 3.0f), 90, 20, juce::Justification::centredLeft);
        graphics.setColour(analyzerMint);
        graphics.drawText(spaceRows[index].second, 1694, juce::roundToInt(y + 3.0f), 156, 20, juce::Justification::centredRight);
    }

    const auto spectrumPanel = juce::Rectangle<float>(40.0f, 632.0f, 1840.0f, 180.0f);
    graphics.setColour(consolePanel.withAlpha(0.9f));
    graphics.fillRoundedRectangle(spectrumPanel, 16.0f);
    graphics.setColour(consoleLine.withAlpha(0.4f));
    graphics.drawRoundedRectangle(spectrumPanel.reduced(0.5f), 16.0f, 1.0f);
    graphics.drawVerticalLine(168, spectrumPanel.getY(), spectrumPanel.getBottom());
    graphics.drawVerticalLine(1670, spectrumPanel.getY(), spectrumPanel.getBottom());
    graphics.setFont(dataFont(10.0f, juce::Font::bold));
    graphics.setColour(consoleText);
    graphics.drawText("LIVE DECAY\nSPECTRUM", 56, 652, 98, 38, juce::Justification::centredLeft);
    graphics.setColour(consoleMuted);
    graphics.setFont(dataFont(9.0f));
    graphics.drawText("EDR / TAIL\nDENSITY\nMODAL BLOOM", 56, 706, 98, 62, juce::Justification::centredLeft);
    graphics.setColour(consoleMuted);
    graphics.setFont(dataFont(10.0f));
    graphics.drawText("EDT", 1692, 660, 44, 20, juce::Justification::centredLeft);
    graphics.drawText("C80", 1692, 696, 44, 20, juce::Justification::centredLeft);
    graphics.drawText("CPU", 1692, 732, 44, 20, juce::Justification::centredLeft);
    graphics.setColour(consoleText);
    graphics.setFont(dataFont(16.0f, juce::Font::bold));
    graphics.drawText("1.84s", 1740, 658, 100, 24, juce::Justification::centredLeft);
    graphics.drawText("-2.7dB", 1740, 694, 100, 24, juce::Justification::centredLeft);
    graphics.drawText("11%", 1740, 730, 100, 24, juce::Justification::centredLeft);

    const auto expert = juce::Rectangle<float>(40.0f, 826.0f, 1840.0f, 170.0f);
    graphics.setColour(consolePanel.withAlpha(0.92f));
    graphics.fillRoundedRectangle(expert, 16.0f);
    graphics.setColour(consoleLine.withAlpha(0.42f));
    graphics.drawRoundedRectangle(expert.reduced(0.5f), 16.0f, 1.0f);
    const std::array<juce::String, 4> tabs{{"FDN & DIFFUSION", "SHIMMER & COLOR", "DYNAMICS & TONE", "SPATIAL & GEOMETRY"}};
    for (size_t index = 0; index < tabs.size(); ++index) {
        const auto x = 40.0f + static_cast<float>(index) * 460.0f;
        if (index == 0) {
            graphics.setColour(analyzerMint.withAlpha(0.07f));
            graphics.fillRect(x, 826.0f, 460.0f, 42.0f);
        }
        graphics.setColour(index == 0 ? analyzerMint : consoleMuted);
        graphics.setFont(dataFont(10.0f, juce::Font::bold));
        graphics.drawText(tabs[index], juce::Rectangle<float>(x, 826.0f, 460.0f, 42.0f), juce::Justification::centred);
    }
    const std::array<std::pair<juce::String, juce::String>, 8> cards{{
        {"LINES", "32"}, {"MATRIX", "Hadamard"}, {"TV RATE", "0.30 Hz"}, {"TV DEPTH", "0.12"},
        {"RAY BLEND", "42%"}, {"WALL LOSS", "0.38"}, {"IMPORT", "DXF"}, {"LATENCY", "11.6 ms"}
    }};
    for (size_t index = 0; index < cards.size(); ++index) {
        const auto x = 54.0f + static_cast<float>(index) * 226.0f;
        drawDataCard(graphics, {x, 880.0f, 212.0f, 100.0f}, cards[index].first, cards[index].second,
                     0.24f + static_cast<float>((index * 13) % 62) / 100.0f);
    }

    graphics.setColour(consoleLine.withAlpha(0.42f));
    graphics.drawHorizontalLine(1020, 40.0f, 1880.0f);
    graphics.setColour(consoleMuted);
    graphics.setFont(dataFont(9.0f));
    graphics.drawText("48 KHZ  ·  64 SAMPLE BLOCK  ·  F32 ENGINE  ·  ZERO-COPY PARAMETER SMOOTHING",
                      60, 1032, 800, 24, juce::Justification::centredLeft);
    graphics.drawText("VERBX v0.8  ·  COLBY LEIDER  ·  AUv3 / VST3",
                      1390, 1032, 450, 24, juce::Justification::centredRight);
}

void VerbXPluginEditor::resized() {
    const auto scale = juce::jmin(
        static_cast<float>(getWidth()) / designWidth,
        static_cast<float>(getHeight()) / designHeight
    );
    const auto offsetX = (static_cast<float>(getWidth()) - designWidth * scale) * 0.5f;
    const auto offsetY = (static_cast<float>(getHeight()) - designHeight * scale) * 0.5f;
    const auto mapBounds = [scale, offsetX, offsetY](juce::Rectangle<float> logical) {
        return juce::Rectangle<int>(
            juce::roundToInt(offsetX + logical.getX() * scale),
            juce::roundToInt(offsetY + logical.getY() * scale),
            juce::roundToInt(logical.getWidth() * scale),
            juce::roundToInt(logical.getHeight() * scale)
        );
    };

    spectrumAnalyzer_.setBounds(mapBounds({170.0f, 642.0f, 1492.0f, 160.0f}));
    for (int index = 0; index < knobCount; ++index) {
        const auto x = 336.0f + static_cast<float>(index) * 99.0f;
        knobLabels_[static_cast<size_t>(index)].setBounds(mapBounds({x, 478.0f, 84.0f, 16.0f}));
        knobs_[static_cast<size_t>(index)].setBounds(mapBounds({x, 495.0f, 84.0f, 102.0f}));
    }
    qualityLabel_.setBounds(mapBounds({1598.0f, 565.0f, 62.0f, 18.0f}));
    qualityBox_.setBounds(mapBounds({1660.0f, 558.0f, 196.0f, 32.0f}));
    freezeButton_.setBounds(mapBounds({1598.0f, 592.0f, 122.0f, 24.0f}));
    reverseButton_.setBounds(mapBounds({1730.0f, 592.0f, 126.0f, 24.0f}));
    rt60Readout_.setBounds(mapBounds({900.0f, 176.0f, 306.0f, 28.0f}));
}
