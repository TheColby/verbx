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

    instance->setNonRealtime(true);
    instance->prepareToPlay(48000.0, 512);
    juce::AudioBuffer<float> buffer(2, 512);
    buffer.clear();
    buffer.setSample(0, 0, 1.0f);
    buffer.setSample(1, 0, 1.0f);
    juce::MidiBuffer midi;
    instance->processBlock(buffer, midi);

    for (int channel = 0; channel < buffer.getNumChannels(); ++channel) {
        for (int sample = 0; sample < buffer.getNumSamples(); ++sample) {
            if (!std::isfinite(buffer.getSample(channel, sample))) {
                return fail("processor produced a non-finite sample");
            }
        }
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
              << editor->getHeight() << "; impulse=finite\n";
    instance->releaseResources();
    return 0;
}
