#include "VerbXPluginProcessor.h"

#include <juce_gui_basics/juce_gui_basics.h>

#include <cmath>
#include <iostream>
#include <memory>

namespace {

int fail(const juce::String& message) {
    std::cerr << "VERBX editor interaction smoke test failed: " << message << '\n';
    return 1;
}

} // namespace

int main(int argc, char* argv[]) {
    juce::ScopedJuceInitialiser_GUI juceInitialiser;
    VerbXPluginProcessor processor;
    processor.prepareToPlay(48000.0, 512);

    std::unique_ptr<juce::AudioProcessorEditor> editor(processor.createEditor());
    if (editor == nullptr) {
        return fail("processor did not create an editor");
    }

    auto* wetSlider = dynamic_cast<juce::Slider*>(editor->findChildWithID("wet"));
    auto* wetParameter = processor.state().getParameter("wet");
    if (wetSlider == nullptr || wetParameter == nullptr) {
        return fail("editor did not expose the Wet dial and parameter");
    }

    wetParameter->setValueNotifyingHost(1.0f);
    const auto sliderBounds = wetSlider->getLookAndFeel()
                                  .getSliderLayout(*wetSlider)
                                  .sliderBounds.toFloat();
    const auto rotary = wetSlider->getRotaryParameters();
    const auto radius = 0.42f * juce::jmin(sliderBounds.getWidth(), sliderBounds.getHeight());
    const auto clickPosition = sliderBounds.getCentre() + juce::Point<float>(
        std::sin(rotary.startAngleRadians) * radius,
        -std::cos(rotary.startAngleRadians) * radius
    );
    const auto eventTime = juce::Time::getCurrentTime();
    const juce::MouseEvent dialClick(
        juce::Desktop::getInstance().getMainMouseSource(),
        clickPosition,
        juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier),
        juce::MouseInputSource::defaultPressure,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        wetSlider,
        wetSlider,
        eventTime,
        clickPosition,
        eventTime,
        1,
        false
    );
    wetSlider->mouseDown(dialClick);
    wetSlider->mouseUp(dialClick);

    if (wetParameter->getValue() > 0.05f) {
        return fail("clicking the Wet dial did not update its parameter");
    }

    auto* expertPageButton = dynamic_cast<juce::Button*>(editor->findChildWithID("page_expert"));
    if (expertPageButton == nullptr) {
        return fail("editor did not expose the Expert page button");
    }
    expertPageButton->setToggleState(true, juce::dontSendNotification);
    expertPageButton->onClick();

    auto* expertWetKnob = dynamic_cast<juce::Slider*>(
        editor->findChildWithID("expert_knob_wet")
    );
    auto* expertWetFader = dynamic_cast<juce::Slider*>(
        editor->findChildWithID("expert_fader_wet")
    );
    if (wetSlider->isVisible()
        || expertWetKnob == nullptr
        || expertWetFader == nullptr
        || !expertWetKnob->isVisible()
        || !expertWetFader->isVisible()) {
        return fail("Expert page did not replace the Perform controls");
    }

    expertWetFader->setValue(
        expertWetFader->getValueFromText("37.5%"),
        juce::sendNotificationSync
    );
    if (std::abs(wetParameter->getValue() - 0.375f) > 0.001f) {
        return fail("Expert percentage entry did not reach host state in musical units");
    }

    auto* expertRt60Fader = dynamic_cast<juce::Slider*>(
        editor->findChildWithID("expert_fader_rt60_coarse")
    );
    if (expertRt60Fader == nullptr) {
        return fail("Expert RT60 precision fader was unavailable");
    }
    expertRt60Fader->setValue(
        expertRt60Fader->getValueFromText("4.8 s"),
        juce::sendNotificationSync
    );
    if (std::abs(processor.effectiveRt60Seconds() - 4.8) > 0.01) {
        return fail("Expert RT60 entry did not invert logarithmic seconds correctly");
    }

    int visibleSelectButtons = 0;
    for (int group = 0; group < 5; ++group) {
        for (int option = 0; option < 4; ++option) {
            auto* button = dynamic_cast<juce::Button*>(editor->findChildWithID(
                "expert_select_" + juce::String(group) + "_" + juce::String(option)
            ));
            if (button != nullptr && button->isVisible()) {
                ++visibleSelectButtons;
            }
        }
    }
    if (visibleSelectButtons != 20) {
        return fail("Expert page did not expose all 20 selector buttons");
    }

    auto* monoWidthButton = dynamic_cast<juce::Button*>(
        editor->findChildWithID("expert_select_1_0")
    );
    auto* widthParameter = processor.state().getParameter("width");
    if (monoWidthButton == nullptr || widthParameter == nullptr) {
        return fail("Expert width selector was unavailable");
    }
    monoWidthButton->setToggleState(true, juce::dontSendNotification);
    monoWidthButton->onClick();
    if (widthParameter->getValue() > 0.001f) {
        return fail("Expert width selector did not update host state");
    }

    if (argc == 2) {
        wetParameter->setValueNotifyingHost(wetParameter->convertTo0to1(0.62f));
        widthParameter->setValueNotifyingHost(widthParameter->convertTo0to1(1.35f));
        auto* rt60Parameter = processor.state().getParameter("rt60_coarse");
        if (rt60Parameter != nullptr) {
            rt60Parameter->setValueNotifyingHost(rt60Parameter->convertTo0to1(0.5f));
        }
        auto* wideWidthButton = dynamic_cast<juce::Button*>(
            editor->findChildWithID("expert_select_1_2")
        );
        if (wideWidthButton != nullptr) {
            wideWidthButton->setToggleState(true, juce::dontSendNotification);
        }
        auto* roomDecayButton = dynamic_cast<juce::Button*>(
            editor->findChildWithID("expert_select_2_1")
        );
        if (roomDecayButton != nullptr) {
            roomDecayButton->setToggleState(true, juce::dontSendNotification);
        }
        const juce::File outputFile(argv[1]);
        outputFile.getParentDirectory().createDirectory();
        outputFile.deleteFile();
        auto stream = outputFile.createOutputStream();
        if (stream == nullptr) {
            return fail("could not create Expert page screenshot output");
        }
        const auto image = editor->createComponentSnapshot(editor->getLocalBounds());
        juce::PNGImageFormat png;
        if (!png.writeImageToStream(image, *stream)) {
            return fail("could not encode Expert page screenshot");
        }
    }

    std::cout << "VERBX dial click and Expert matrix reached host state; selectors="
              << visibleSelectButtons << '\n';
    processor.releaseResources();
    return 0;
}
