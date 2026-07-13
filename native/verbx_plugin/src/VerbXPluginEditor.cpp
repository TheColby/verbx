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

constexpr std::array<std::array<const char*, 4>, 5> expertSelectLabels{{
    {{"HOST", "2X", "4X", "192K"}},
    {{"MONO", "NAT", "WIDE", "ULTRA"}},
    {{"TIGHT", "ROOM", "HALL", "FREEZE"}},
    {{"DRY", "INSERT", "PAR", "SEND"}},
    {{"CLEAN", "WARM", "DARK", "AIR"}},
}};

constexpr std::array<const char*, 5> expertSelectGroupLabels{{
    "QUALITY", "WIDTH MATRIX", "DECAY RANGE", "MIX ROUTING", "TAIL CHARACTER"
}};

constexpr std::array<const char*, 5> expertSelectTooltips{{
    "Select the host-visible internal quality policy.",
    "Write a calibrated stereo-width value.",
    "Write logarithmic RT60 and Freeze state.",
    "Write a matched Dry/Wet routing pair.",
    "Write a matched Damping/Diffusion character pair.",
}};

void configureParameterText(juce::Slider& slider, int index, bool precision) {
    if (index == 0) {
        slider.textFromValueFunction = [precision](double value) {
            return juce::String(value, precision ? 2 : 1) + " ms";
        };
        slider.valueFromTextFunction = [](const juce::String& text) {
            return text.getDoubleValue();
        };
    } else if (index == 2) {
        slider.textFromValueFunction = [](double value) {
            const auto seconds = verbx_plugin_map_rt60_seconds(value, 0.0);
            const auto decimals = seconds < 1.0 ? 3 : (seconds < 10.0 ? 2 : 1);
            return juce::String(seconds, decimals) + " s";
        };
        slider.valueFromTextFunction = [](const juce::String& text) {
            const auto seconds = juce::jlimit(0.01, 360.0, text.getDoubleValue());
            return std::log(seconds / 0.01) / std::log(360.0 / 0.01);
        };
    } else if (index == 3) {
        slider.textFromValueFunction = [precision](double value) {
            const auto percent = value * 20.0;
            return juce::String(percent >= 0.0 ? "+" : "")
                + juce::String(percent, precision ? 1 : 0) + "%";
        };
        slider.valueFromTextFunction = [](const juce::String& text) {
            return text.getDoubleValue() / 20.0;
        };
    } else {
        slider.textFromValueFunction = [precision](double value) {
            return juce::String(value * 100.0, precision ? 1 : 0) + "%";
        };
        slider.valueFromTextFunction = [](const juce::String& text) {
            return text.getDoubleValue() / 100.0;
        };
    }
    slider.updateText();
}

} // namespace

void VerbXKnobSlider::mouseDown(const juce::MouseEvent& event) {
    if (isEnabled()
        && event.mods.isLeftButtonDown()
        && !event.mods.isPopupMenu()
        && event.getNumberOfClicks() == 1) {
        setValueFromDialPosition(event.position);
    }

    juce::Slider::mouseDown(event);
}

void VerbXKnobSlider::setValueFromDialPosition(juce::Point<float> position) {
    const auto dialBounds = getLookAndFeel().getSliderLayout(*this).sliderBounds.toFloat();
    if (!dialBounds.contains(position)) {
        return;
    }

    const auto offset = position - dialBounds.getCentre();
    if (offset.getDistanceSquaredFromOrigin() <= 25.0f) {
        return;
    }

    const auto rotary = getRotaryParameters();
    auto angle = std::atan2(static_cast<double>(offset.x), static_cast<double>(-offset.y));
    while (angle < rotary.startAngleRadians) {
        angle += juce::MathConstants<double>::twoPi;
    }

    if (angle > rotary.endAngleRadians) {
        const auto distanceFromStart = std::abs(std::remainder(
            angle - static_cast<double>(rotary.startAngleRadians),
            juce::MathConstants<double>::twoPi
        ));
        const auto distanceFromEnd = std::abs(std::remainder(
            angle - static_cast<double>(rotary.endAngleRadians),
            juce::MathConstants<double>::twoPi
        ));
        angle = distanceFromStart <= distanceFromEnd
            ? rotary.startAngleRadians
            : rotary.endAngleRadians;
    }

    const auto proportion = juce::jlimit(
        0.0,
        1.0,
        (angle - rotary.startAngleRadians)
            / (rotary.endAngleRadians - rotary.startAngleRadians)
    );
    setValue(proportionOfLengthToValue(proportion), juce::sendNotificationSync);
}

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
    : AudioProcessorEditor(&processor),
      processor_(processor),
      spectrumAnalyzer_(processor),
      tooltipWindow_(this, 650) {
    addAndMakeVisible(spectrumAnalyzer_);
    configureControls();
    configureExpertControls();
    setResizable(true, true);
    // Hosts wrap editors in differently constrained windows. Keep the console
    // responsive instead of forcing a large, fixed-ratio surface on the host.
    setResizeLimits(800, 450, 2560, 1440);
    setSize(1280, 720);
    startTimerHz(15);
}

void VerbXPluginEditor::configureControls() {
    auto& state = processor_.state();
    for (auto* button : {&performPageButton_, &expertPageButton_}) {
        button->setClickingTogglesState(true);
        button->setRadioGroupId(9001);
        button->setColour(juce::TextButton::buttonColourId, consolePanel);
        button->setColour(juce::TextButton::buttonOnColourId, analyzerMint);
        button->setColour(juce::TextButton::textColourOffId, consoleMuted);
        button->setColour(juce::TextButton::textColourOnId, analyzerInk);
        addAndMakeVisible(*button);
    }
    performPageButton_.setComponentID("page_perform");
    expertPageButton_.setComponentID("page_expert");
    performPageButton_.setTooltip("Open the visual performance console.");
    expertPageButton_.setTooltip("Open linked precision controls and selector macros.");
    performPageButton_.setToggleState(true, juce::dontSendNotification);
    performPageButton_.onClick = [this] {
        activePage_ = Page::perform;
        updatePageVisibility();
        resized();
        repaint();
    };
    expertPageButton_.onClick = [this] {
        activePage_ = Page::expert;
        updatePageVisibility();
        resized();
        repaint();
    };

    for (int index = 0; index < knobCount; ++index) {
        const auto& definition = knobDefinitions[static_cast<size_t>(index)];
        auto& knob = knobs_[static_cast<size_t>(index)];
        auto& label = knobLabels_[static_cast<size_t>(index)];
        knob.setLookAndFeel(&lookAndFeel_);
        knob.setSliderStyle(juce::Slider::RotaryVerticalDrag);
        knob.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 84, 20);
        knob.setMouseDragSensitivity(180);
        knob.setScrollWheelEnabled(true);
        knob.setPopupMenuEnabled(true);
        knob.setPopupDisplayEnabled(true, true, this, 1200);
        knob.setMouseCursor(juce::MouseCursor::PointingHandCursor);
        knob.setName(definition.label);
        knob.setTitle(definition.label);
        knob.setComponentID(definition.parameterId);
        knob.setTooltip("Click the dial or drag vertically; scroll to adjust; double-click to reset.");
        if (const auto* parameter = state.getParameter(definition.parameterId)) {
            knob.setDoubleClickReturnValue(
                true,
                parameter->convertFrom0to1(parameter->getDefaultValue())
            );
        }
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
        configureParameterText(knob, index, false);
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

void VerbXPluginEditor::configureExpertControls() {
    auto& state = processor_.state();
    for (int index = 0; index < knobCount; ++index) {
        const auto& definition = knobDefinitions[static_cast<size_t>(index)];
        auto& knob = expertKnobs_[static_cast<size_t>(index)];
        auto& knobLabel = expertKnobLabels_[static_cast<size_t>(index)];
        auto& fader = expertFaders_[static_cast<size_t>(index)];
        auto& faderLabel = expertFaderLabels_[static_cast<size_t>(index)];

        knob.setLookAndFeel(&lookAndFeel_);
        knob.setSliderStyle(juce::Slider::RotaryVerticalDrag);
        knob.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 108, 20);
        knob.setMouseDragSensitivity(220);
        knob.setScrollWheelEnabled(true);
        knob.setPopupMenuEnabled(true);
        knob.setPopupDisplayEnabled(true, true, this, 1200);
        knob.setMouseCursor(juce::MouseCursor::PointingHandCursor);
        knob.setName(juce::String("Expert ") + definition.label);
        knob.setTitle(juce::String("Expert ") + definition.label);
        knob.setComponentID(juce::String("expert_knob_") + definition.parameterId);
        knob.setTooltip("Click the arc or drag vertically; double-click restores the parameter default.");
        knob.setColour(juce::Slider::rotarySliderFillColourId, analyzerGold);
        knob.setColour(juce::Slider::rotarySliderOutlineColourId, analyzerMint.withAlpha(0.20f));
        knob.setColour(juce::Slider::thumbColourId, analyzerMint);
        knob.setColour(juce::Slider::textBoxTextColourId, consoleText);
        knob.setColour(juce::Slider::textBoxBackgroundColourId, analyzerInk.withAlpha(0.78f));
        knob.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
        if (const auto* parameter = state.getParameter(definition.parameterId)) {
            knob.setDoubleClickReturnValue(
                true,
                parameter->convertFrom0to1(parameter->getDefaultValue())
            );
        }
        addAndMakeVisible(knob);

        knobLabel.setText(definition.label, juce::dontSendNotification);
        knobLabel.setJustificationType(juce::Justification::centred);
        knobLabel.setFont(dataFont(9.5f, juce::Font::bold));
        knobLabel.setColour(juce::Label::textColourId, analyzerGold);
        addAndMakeVisible(knobLabel);
        expertKnobAttachments_[static_cast<size_t>(index)] = std::make_unique<SliderAttachment>(
            state,
            definition.parameterId,
            knob
        );
        configureParameterText(knob, index, true);

        fader.setSliderStyle(juce::Slider::LinearHorizontal);
        fader.setTextBoxStyle(juce::Slider::TextBoxRight, false, 92, 24);
        fader.setMouseDragSensitivity(420);
        fader.setScrollWheelEnabled(true);
        fader.setPopupDisplayEnabled(true, false, this, 1000);
        fader.setComponentID(juce::String("expert_fader_") + definition.parameterId);
        fader.setName(juce::String("Precision ") + definition.label);
        fader.setTitle(juce::String("Precision ") + definition.label);
        fader.setTooltip("Precision fader for the same automatable host parameter.");
        fader.setColour(juce::Slider::trackColourId, analyzerMint.withAlpha(0.82f));
        fader.setColour(juce::Slider::thumbColourId, analyzerGold);
        fader.setColour(juce::Slider::backgroundColourId, juce::Colours::white.withAlpha(0.08f));
        fader.setColour(juce::Slider::textBoxTextColourId, consoleText);
        fader.setColour(juce::Slider::textBoxBackgroundColourId, analyzerInk.withAlpha(0.78f));
        fader.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
        addAndMakeVisible(fader);

        faderLabel.setText(definition.label, juce::dontSendNotification);
        faderLabel.setJustificationType(juce::Justification::centredLeft);
        faderLabel.setFont(dataFont(9.0f, juce::Font::bold));
        faderLabel.setColour(juce::Label::textColourId, consoleMuted);
        addAndMakeVisible(faderLabel);
        expertFaderAttachments_[static_cast<size_t>(index)] = std::make_unique<SliderAttachment>(
            state,
            definition.parameterId,
            fader
        );
        configureParameterText(fader, index, true);
    }

    for (int group = 0; group < expertSelectGroupCount; ++group) {
        for (int option = 0; option < expertSelectsPerGroup; ++option) {
            const auto index = group * expertSelectsPerGroup + option;
            auto& button = expertSelectButtons_[static_cast<size_t>(index)];
            button.setButtonText(expertSelectLabels[static_cast<size_t>(group)][static_cast<size_t>(option)]);
            button.setClickingTogglesState(true);
            button.setRadioGroupId(9100 + group);
            button.setComponentID("expert_select_" + juce::String(group) + "_" + juce::String(option));
            button.setTooltip(expertSelectTooltips[static_cast<size_t>(group)]);
            button.setColour(juce::TextButton::buttonColourId, consolePanel.brighter(0.06f));
            button.setColour(juce::TextButton::buttonOnColourId, analyzerGold);
            button.setColour(juce::TextButton::textColourOffId, consoleMuted);
            button.setColour(juce::TextButton::textColourOnId, analyzerInk);
            button.onClick = [this, group, option] { selectExpertMacro(group, option); };
            addAndMakeVisible(button);
        }
    }
    syncExpertMacroSelections();
    updatePageVisibility();
}

void VerbXPluginEditor::setPlainParameter(const char* parameterId, float value) {
    if (auto* parameter = processor_.state().getParameter(parameterId)) {
        parameter->beginChangeGesture();
        parameter->setValueNotifyingHost(parameter->convertTo0to1(value));
        parameter->endChangeGesture();
    }
}

float VerbXPluginEditor::plainParameter(const char* parameterId) const {
    const auto* value = processor_.state().getRawParameterValue(parameterId);
    return value != nullptr ? value->load(std::memory_order_relaxed) : 0.0f;
}

void VerbXPluginEditor::selectExpertMacro(int group, int option) {
    if (group == 0) {
        setPlainParameter("quality_mode", static_cast<float>(option));
    } else if (group == 1) {
        constexpr std::array<float, 4> values{0.0f, 1.0f, 1.35f, 2.0f};
        setPlainParameter("width", values[static_cast<size_t>(option)]);
    } else if (group == 2) {
        constexpr std::array<double, 4> seconds{0.35, 1.9, 4.8, 360.0};
        const auto normalized = static_cast<float>(
            std::log(seconds[static_cast<size_t>(option)] / 0.01) / std::log(360.0 / 0.01)
        );
        setPlainParameter("rt60_coarse", normalized);
        setPlainParameter("freeze", option == 3 ? 1.0f : 0.0f);
    } else if (group == 3) {
        constexpr std::array<float, 4> dry{1.0f, 0.78f, 1.0f, 0.0f};
        constexpr std::array<float, 4> wet{0.0f, 0.62f, 1.0f, 1.0f};
        setPlainParameter("dry", dry[static_cast<size_t>(option)]);
        setPlainParameter("wet", wet[static_cast<size_t>(option)]);
    } else if (group == 4) {
        constexpr std::array<float, 4> damping{0.41f, 0.48f, 0.78f, 0.12f};
        constexpr std::array<float, 4> diffusion{0.65f, 0.72f, 0.75f, 0.55f};
        setPlainParameter("damping", damping[static_cast<size_t>(option)]);
        setPlainParameter("diffusion", diffusion[static_cast<size_t>(option)]);
    }
}

void VerbXPluginEditor::syncExpertMacroSelections() {
    const auto setGroup = [this](int group, int selectedOption) {
        for (int option = 0; option < expertSelectsPerGroup; ++option) {
            const auto index = group * expertSelectsPerGroup + option;
            expertSelectButtons_[static_cast<size_t>(index)].setToggleState(
                option == selectedOption,
                juce::dontSendNotification
            );
        }
    };
    const auto matchingOption = [](float value, const auto& values, float tolerance) {
        for (int option = 0; option < static_cast<int>(values.size()); ++option) {
            if (std::abs(value - static_cast<float>(values[static_cast<size_t>(option)])) <= tolerance) {
                return option;
            }
        }
        return -1;
    };

    setGroup(0, juce::jlimit(0, 3, juce::roundToInt(plainParameter("quality_mode"))));

    constexpr std::array<float, 4> widths{0.0f, 1.0f, 1.35f, 2.0f};
    setGroup(1, matchingOption(plainParameter("width"), widths, 0.001f));

    if (plainParameter("freeze") >= 0.5f) {
        setGroup(2, 3);
    } else {
        constexpr std::array<double, 4> seconds{0.35, 1.9, 4.8, 360.0};
        std::array<float, 4> coarse{};
        for (size_t index = 0; index < seconds.size(); ++index) {
            coarse[index] = static_cast<float>(
                std::log(seconds[index] / 0.01) / std::log(360.0 / 0.01)
            );
        }
        setGroup(2, matchingOption(plainParameter("rt60_coarse"), coarse, 0.001f));
    }

    constexpr std::array<float, 4> dry{1.0f, 0.78f, 1.0f, 0.0f};
    constexpr std::array<float, 4> wet{0.0f, 0.62f, 1.0f, 1.0f};
    auto mixOption = -1;
    for (int option = 0; option < 4; ++option) {
        if (std::abs(plainParameter("dry") - dry[static_cast<size_t>(option)]) <= 0.001f
            && std::abs(plainParameter("wet") - wet[static_cast<size_t>(option)]) <= 0.001f) {
            mixOption = option;
            break;
        }
    }
    setGroup(3, mixOption);

    constexpr std::array<float, 4> damping{0.41f, 0.48f, 0.78f, 0.12f};
    constexpr std::array<float, 4> diffusion{0.65f, 0.72f, 0.75f, 0.55f};
    auto characterOption = -1;
    for (int option = 0; option < 4; ++option) {
        if (std::abs(plainParameter("damping") - damping[static_cast<size_t>(option)]) <= 0.001f
            && std::abs(plainParameter("diffusion") - diffusion[static_cast<size_t>(option)]) <= 0.001f) {
            characterOption = option;
            break;
        }
    }
    setGroup(4, characterOption);
}

void VerbXPluginEditor::updatePageVisibility() {
    const auto performVisible = activePage_ == Page::perform;
    for (int index = 0; index < knobCount; ++index) {
        knobs_[static_cast<size_t>(index)].setVisible(performVisible);
        knobLabels_[static_cast<size_t>(index)].setVisible(performVisible);
        expertKnobs_[static_cast<size_t>(index)].setVisible(!performVisible);
        expertKnobLabels_[static_cast<size_t>(index)].setVisible(!performVisible);
        expertFaders_[static_cast<size_t>(index)].setVisible(!performVisible);
        expertFaderLabels_[static_cast<size_t>(index)].setVisible(!performVisible);
    }
    freezeButton_.setVisible(performVisible);
    reverseButton_.setVisible(performVisible);
    qualityBox_.setVisible(performVisible);
    qualityLabel_.setVisible(performVisible);
    for (auto& button : expertSelectButtons_) {
        button.setVisible(!performVisible);
    }
}

void VerbXPluginEditor::timerCallback() {
    const auto seconds = processor_.effectiveRt60Seconds();
    const auto precision = seconds < 1.0 ? 3 : (seconds < 10.0 ? 2 : 1);
    rt60Readout_.setText(
        "EFFECTIVE RT60  " + juce::String(seconds, precision) + " s",
        juce::dontSendNotification
    );
    syncExpertMacroSelections();
}

void VerbXPluginEditor::paintExpertPage(juce::Graphics& graphics) {
    graphics.setColour(consoleText);
    graphics.setFont(consoleFont(24.0f, juce::Font::bold));
    graphics.drawText("EXPERT CONTROL MATRIX", 54, 112, 420, 34, juce::Justification::centredLeft);
    graphics.setColour(consoleMuted);
    graphics.setFont(consoleFont(12.0f));
    graphics.drawText(
        "Nine automatable dials, nine linked precision faders, and twenty performance selectors",
        470,
        117,
        900,
        24,
        juce::Justification::centredLeft
    );
    for (int index = 0; index < knobCount; ++index) {
        const auto x = 48.0f + static_cast<float>(index) * 204.0f;
        const auto card = juce::Rectangle<float>(x, 154.0f, 190.0f, 176.0f);
        graphics.setColour(juce::Colour::fromRGB(15, 22, 26));
        graphics.fillRoundedRectangle(card, 14.0f);
        graphics.setColour((index % 3 == 0 ? analyzerGold : analyzerMint).withAlpha(0.22f));
        graphics.drawRoundedRectangle(card.reduced(0.5f), 14.0f, 1.0f);
        graphics.setColour(consoleMuted.withAlpha(0.55f));
        graphics.setFont(dataFont(7.5f, juce::Font::bold));
        graphics.drawText(
            "P" + juce::String(index + 1).paddedLeft('0', 2),
            juce::roundToInt(x + 12.0f),
            302,
            36,
            14,
            juce::Justification::centredLeft
        );
    }

    const auto analyzerPanel = juce::Rectangle<float>(48.0f, 344.0f, 1824.0f, 126.0f);
    drawPanel(graphics, analyzerPanel, "REALTIME SPECTRUM / TAIL ENERGY", "20 HZ - 20 KHZ");

    for (int index = 0; index < knobCount; ++index) {
        const auto column = index % 3;
        const auto row = index / 3;
        const auto x = 48.0f + static_cast<float>(column) * 608.0f;
        const auto y = 490.0f + static_cast<float>(row) * 104.0f;
        const auto card = juce::Rectangle<float>(x, y, 592.0f, 88.0f);
        graphics.setColour(juce::Colour::fromRGB(14, 21, 25));
        graphics.fillRoundedRectangle(card, 12.0f);
        graphics.setColour(consoleLine.withAlpha(0.28f));
        graphics.drawRoundedRectangle(card.reduced(0.5f), 12.0f, 1.0f);
        graphics.setColour(analyzerMint.withAlpha(0.13f));
        graphics.fillRoundedRectangle(x + 14.0f, y + 57.0f, 554.0f, 4.0f, 2.0f);
    }

    for (int group = 0; group < expertSelectGroupCount; ++group) {
        const auto x = 48.0f + static_cast<float>(group) * 366.0f;
        const auto bank = juce::Rectangle<float>(x, 826.0f, 350.0f, 148.0f);
        graphics.setColour(juce::Colour::fromRGB(17, 24, 28));
        graphics.fillRoundedRectangle(bank, 14.0f);
        graphics.setColour(analyzerGold.withAlpha(0.24f));
        graphics.drawRoundedRectangle(bank.reduced(0.5f), 14.0f, 1.0f);
        graphics.setColour(consoleMuted);
        graphics.setFont(dataFont(9.0f, juce::Font::bold));
        graphics.drawText(
            expertSelectGroupLabels[static_cast<size_t>(group)],
            juce::roundToInt(x + 14.0f),
            842,
            322,
            20,
            juce::Justification::centredLeft
        );
        graphics.setColour(consoleMuted.withAlpha(0.72f));
        graphics.setFont(consoleFont(10.0f));
        graphics.drawText(
            group == 0 ? "Internal render policy"
                       : group == 1 ? "Stereo field target"
                                    : group == 2 ? "Logarithmic RT60 macro"
                                                 : group == 3 ? "Dry / wet gain topology"
                                                              : "Damping + diffusion pair",
            juce::roundToInt(x + 14.0f),
            942,
            322,
            18,
            juce::Justification::centredLeft
        );
    }

    graphics.setColour(consoleMuted);
    graphics.setFont(dataFont(9.0f));
    graphics.drawText(
        "TIP  CLICK A DIAL ARC, DRAG VERTICALLY, SCROLL, OR TYPE A VALUE. DOUBLE-CLICK RESETS.",
        54,
        998,
        1200,
        24,
        juce::Justification::centredLeft
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

    graphics.setColour(consolePanel);
    graphics.fillRoundedRectangle(1762.0f, 37.0f, 90.0f, 38.0f, 18.0f);
    graphics.setColour(analyzerMint);
    graphics.fillEllipse(1774.0f, 52.0f, 8.0f, 8.0f);
    graphics.setFont(dataFont(10.0f, juce::Font::bold));
    graphics.drawText("LIVE", 1786, 44, 54, 22, juce::Justification::centredLeft);

    if (activePage_ == Page::expert) {
        paintExpertPage(graphics);
        return;
    }

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

    performPageButton_.setBounds(mapBounds({1538.0f, 37.0f, 100.0f, 38.0f}));
    expertPageButton_.setBounds(mapBounds({1646.0f, 37.0f, 100.0f, 38.0f}));
    rt60Readout_.setBounds(mapBounds({900.0f, 176.0f, 306.0f, 28.0f}));

    if (activePage_ == Page::expert) {
        rt60Readout_.setBounds(mapBounds({1380.0f, 112.0f, 460.0f, 28.0f}));
        spectrumAnalyzer_.setBounds(mapBounds({62.0f, 378.0f, 1796.0f, 78.0f}));
        for (int index = 0; index < knobCount; ++index) {
            const auto knobX = 63.0f + static_cast<float>(index) * 204.0f;
            expertKnobLabels_[static_cast<size_t>(index)].setBounds(
                mapBounds({knobX, 166.0f, 160.0f, 18.0f})
            );
            expertKnobs_[static_cast<size_t>(index)].setBounds(
                mapBounds({knobX, 184.0f, 160.0f, 130.0f})
            );

            const auto column = index % 3;
            const auto row = index / 3;
            const auto faderX = 68.0f + static_cast<float>(column) * 608.0f;
            const auto faderY = 500.0f + static_cast<float>(row) * 104.0f;
            expertFaderLabels_[static_cast<size_t>(index)].setBounds(
                mapBounds({faderX, faderY, 300.0f, 18.0f})
            );
            expertFaders_[static_cast<size_t>(index)].setBounds(
                mapBounds({faderX, faderY + 22.0f, 552.0f, 42.0f})
            );
        }
        for (int group = 0; group < expertSelectGroupCount; ++group) {
            const auto bankX = 62.0f + static_cast<float>(group) * 366.0f;
            for (int option = 0; option < expertSelectsPerGroup; ++option) {
                const auto index = group * expertSelectsPerGroup + option;
                expertSelectButtons_[static_cast<size_t>(index)].setBounds(mapBounds({
                    bankX + static_cast<float>(option) * 82.0f,
                    878.0f,
                    76.0f,
                    42.0f,
                }));
            }
        }
        return;
    }

    spectrumAnalyzer_.setBounds(mapBounds({170.0f, 642.0f, 1492.0f, 160.0f}));
    for (int index = 0; index < knobCount; ++index) {
        const auto x = 336.0f + static_cast<float>(index) * 99.0f;
        knobLabels_[static_cast<size_t>(index)].setBounds(mapBounds({x, 478.0f, 84.0f, 16.0f}));
        knobs_[static_cast<size_t>(index)].setBounds(mapBounds({x - 4.0f, 491.0f, 92.0f, 110.0f}));
    }
    qualityLabel_.setBounds(mapBounds({1598.0f, 565.0f, 62.0f, 18.0f}));
    qualityBox_.setBounds(mapBounds({1660.0f, 558.0f, 196.0f, 32.0f}));
    freezeButton_.setBounds(mapBounds({1598.0f, 592.0f, 122.0f, 24.0f}));
    reverseButton_.setBounds(mapBounds({1730.0f, 592.0f, 126.0f, 24.0f}));
}
