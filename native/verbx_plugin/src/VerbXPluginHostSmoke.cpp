#include <juce_audio_utils/juce_audio_utils.h>

#include <cmath>
#include <iostream>

namespace {

int fail(const juce::String& message) {
    std::cerr << "VERBX host smoke test failed: " << message << '\n';
    return 1;
}

juce::AudioPluginFormat* findFormat(
    juce::AudioPluginFormatManager& manager,
    const juce::String& formatName
) {
    for (auto* format : manager.getFormats()) {
        if (format->getName().equalsIgnoreCase(formatName)) {
            return format;
        }
    }
    return nullptr;
}

bool setNormalizedParameter(
    juce::AudioProcessor& processor,
    const juce::String& parameterName,
    float normalizedValue
) {
    for (auto* parameter : processor.getParameters()) {
        if (parameter->getName(128).equalsIgnoreCase(parameterName)) {
            parameter->setValueNotifyingHost(normalizedValue);
            return true;
        }
    }
    return false;
}

} // namespace

int main(int argc, char* argv[]) {
    juce::ScopedJuceInitialiser_GUI juceInitialiser;

    if (argc != 3) {
        std::cerr << "Usage: VERBXPluginHostSmoke <AudioUnit|VST3> <path-or-identifier>\n";
        return 2;
    }

    const juce::String formatName(argv[1]);
    const juce::String pluginIdentifier(argv[2]);
    juce::AudioPluginFormatManager manager;
    manager.addDefaultFormats();

    auto* format = findFormat(manager, formatName);
    if (format == nullptr) {
        return fail("host format is unavailable: " + formatName);
    }

    juce::OwnedArray<juce::PluginDescription> descriptions;
    format->findAllTypesForFile(descriptions, pluginIdentifier);
    if (descriptions.isEmpty()) {
        return fail("no plug-in types found at " + pluginIdentifier);
    }

    juce::String error;
    auto instance = format->createInstanceFromDescription(
        *descriptions[0],
        48000.0,
        512,
        error
    );
    if (instance == nullptr) {
        return fail("instantiation failed: " + error);
    }

    auto layout = instance->getBusesLayout();
    if (layout.inputBuses.isEmpty() || layout.outputBuses.isEmpty()) {
        return fail("processor did not expose main input/output buses");
    }
    layout.inputBuses.set(0, juce::AudioChannelSet::stereo());
    layout.outputBuses.set(0, juce::AudioChannelSet::stereo());
    if (!instance->setBusesLayout(layout)) {
        return fail("processor rejected a stereo host layout");
    }

    instance->setNonRealtime(true);
    instance->prepareToPlay(48000.0, 512);
    if (!setNormalizedParameter(*instance, "Pre-Delay", 0.0f)
        || !setNormalizedParameter(*instance, "Dry", 0.0f)
        || !setNormalizedParameter(*instance, "Wet", 1.0f)) {
        return fail("required wet-tail parameters were unavailable");
    }

    juce::AudioBuffer<float> buffer(2, 512);
    juce::MidiBuffer midi;
    double wetEnergy = 0.0;
    for (int block = 0; block < 64; ++block) {
        buffer.clear();
        if (block == 0) {
            buffer.setSample(0, 0, 1.0f);
            buffer.setSample(1, 0, 1.0f);
        }
        instance->processBlock(buffer, midi);
        for (int channel = 0; channel < buffer.getNumChannels(); ++channel) {
            for (int sample = 0; sample < buffer.getNumSamples(); ++sample) {
                const auto value = buffer.getSample(channel, sample);
                if (!std::isfinite(value)) {
                    return fail("processor produced a non-finite sample");
                }
                wetEnergy += std::abs(static_cast<double>(value));
            }
        }
    }
    if (wetEnergy <= 0.01) {
        return fail("processor did not produce an audible wet tail");
    }

    std::unique_ptr<juce::AudioProcessorEditor> editor(instance->createEditorIfNeeded());
    if (editor == nullptr) {
        return fail("processor did not create an editor");
    }
    if (editor->getWidth() > 1280 || editor->getHeight() > 720) {
        return fail("editor exceeded its host-safe opening size");
    }

    std::cout << descriptions[0]->name << " " << format->getName()
              << " loaded; editor=" << editor->getWidth() << "x"
              << editor->getHeight() << "; wet-energy=" << wetEnergy << '\n';
    instance->releaseResources();
    return 0;
}
