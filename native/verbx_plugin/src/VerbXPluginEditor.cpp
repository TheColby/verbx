#include "VerbXPluginEditor.h"
#include "VerbXPluginProcessor.h"

VerbXPluginEditor::VerbXPluginEditor(VerbXPluginProcessor& processor)
    : AudioProcessorEditor(&processor), processor_(processor) {
    setResizable(true, true);
    setResizeLimits(960, 600, 2560, 1600);
    setSize(1560, 920);
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

    graphics.setColour(juce::Colour::fromRGB(238, 247, 244));
    graphics.setFont(juce::FontOptions(22.0f, juce::Font::bold));
    graphics.drawText(
        "Full-screen spatial console scaffold",
        getLocalBounds().reduced(48, 160),
        juce::Justification::centred
    );
}

void VerbXPluginEditor::resized() {
    juce::ignoreUnused(processor_);
}
