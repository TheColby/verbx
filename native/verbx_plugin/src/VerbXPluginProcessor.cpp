#include "VerbXPluginProcessor.h"
#include "VerbXPluginEditor.h"

#include <memory>
#include <thread>
#include <vector>

namespace {

juce::String parameterId(verbx_plugin_parameter_id id) {
    const auto* parameter = verbx_plugin_parameter_by_id(id);
    return parameter != nullptr ? juce::String(parameter->key) : juce::String();
}

float parameterValue(const std::atomic<float>* value) {
    return value != nullptr ? value->load(std::memory_order_relaxed) : 0.0f;
}

juce::ParameterID versionedParameterId(const verbx_plugin_parameter& parameter) {
    return {parameter.key, 1};
}

} // namespace

VerbXPluginProcessor::VerbXPluginProcessor()
    : AudioProcessor(BusesProperties().withInput("Input", juce::AudioChannelSet::stereo(), true)
                                      .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      parameters_(*this, nullptr, "VERBX", createParameterLayout()) {
    cacheParameterPointers();
    requestedQualityMode_.store(
        juce::roundToInt(parameterValue(parameterPointers_.qualityMode)),
        std::memory_order_relaxed
    );
    parameters_.addParameterListener(parameterId(VERBX_PLUGIN_PARAM_QUALITY_MODE), this);
}

VerbXPluginProcessor::~VerbXPluginProcessor() {
    parameters_.removeParameterListener(parameterId(VERBX_PLUGIN_PARAM_QUALITY_MODE), this);
    cancelPendingUpdate();
    acquireRealtimeContextGuard();
    verbx_plugin_realtime_release(&realtimeContext_);
    releaseRealtimeContextGuard();
}

juce::AudioProcessorValueTreeState::ParameterLayout VerbXPluginProcessor::createParameterLayout() {
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> layout;
    layout.reserve(verbx_plugin_parameter_count());

    for (size_t index = 0; index < verbx_plugin_parameter_count(); ++index) {
        const auto* parameter = verbx_plugin_parameter_at(index);
        if (parameter == nullptr) {
            continue;
        }

        const auto id = versionedParameterId(*parameter);
        const juce::String label(parameter->label);
        if (parameter->kind == VERBX_PLUGIN_PARAMETER_BOOL) {
            layout.push_back(std::make_unique<juce::AudioParameterBool>(
                id,
                label,
                parameter->default_value >= 0.5
            ));
        } else if (parameter->kind == VERBX_PLUGIN_PARAMETER_CHOICE) {
            juce::StringArray choices{"Host", "2x", "4x", "Target 192 kHz"};
            layout.push_back(std::make_unique<juce::AudioParameterChoice>(
                id,
                label,
                choices,
                static_cast<int>(parameter->default_value)
            ));
        } else {
            layout.push_back(std::make_unique<juce::AudioParameterFloat>(
                id,
                label,
                juce::NormalisableRange<float>(
                    static_cast<float>(parameter->minimum),
                    static_cast<float>(parameter->maximum)
                ),
                static_cast<float>(parameter->default_value)
            ));
        }
    }

    return {layout.begin(), layout.end()};
}

void VerbXPluginProcessor::cacheParameterPointers() {
    parameterPointers_.preDelayMs = parameters_.getRawParameterValue(
        parameterId(VERBX_PLUGIN_PARAM_PRE_DELAY_MS)
    );
    parameterPointers_.roomSize = parameters_.getRawParameterValue(
        parameterId(VERBX_PLUGIN_PARAM_ROOM_SIZE)
    );
    parameterPointers_.rt60Coarse = parameters_.getRawParameterValue(
        parameterId(VERBX_PLUGIN_PARAM_RT60_COARSE)
    );
    parameterPointers_.rt60Fine = parameters_.getRawParameterValue(
        parameterId(VERBX_PLUGIN_PARAM_RT60_FINE)
    );
    parameterPointers_.damping = parameters_.getRawParameterValue(
        parameterId(VERBX_PLUGIN_PARAM_DAMPING)
    );
    parameterPointers_.width = parameters_.getRawParameterValue(
        parameterId(VERBX_PLUGIN_PARAM_WIDTH)
    );
    parameterPointers_.diffusion = parameters_.getRawParameterValue(
        parameterId(VERBX_PLUGIN_PARAM_DIFFUSION)
    );
    parameterPointers_.wet = parameters_.getRawParameterValue(
        parameterId(VERBX_PLUGIN_PARAM_WET)
    );
    parameterPointers_.dry = parameters_.getRawParameterValue(
        parameterId(VERBX_PLUGIN_PARAM_DRY)
    );
    parameterPointers_.freeze = parameters_.getRawParameterValue(
        parameterId(VERBX_PLUGIN_PARAM_FREEZE)
    );
    parameterPointers_.reverse = parameters_.getRawParameterValue(
        parameterId(VERBX_PLUGIN_PARAM_REVERSE)
    );
    parameterPointers_.qualityMode = parameters_.getRawParameterValue(
        parameterId(VERBX_PLUGIN_PARAM_QUALITY_MODE)
    );
}

void VerbXPluginProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
    analyzerSampleRate_.store(sampleRate, std::memory_order_release);
    analyzerReadPosition_.store(0U, std::memory_order_relaxed);
    analyzerWritePosition_.store(0U, std::memory_order_release);

    const auto qualityMode = juce::jlimit(
        static_cast<int>(VERBX_PLUGIN_QUALITY_HOST),
        static_cast<int>(VERBX_PLUGIN_QUALITY_TARGET_192K),
        juce::roundToInt(parameterValue(parameterPointers_.qualityMode))
    );
    preparedHostSampleRate_.store(sampleRate, std::memory_order_release);
    preparedBlockSize_.store(juce::jmax(samplesPerBlock, 1), std::memory_order_release);
    requestedQualityMode_.store(qualityMode, std::memory_order_release);
    prepareRealtimeContext(sampleRate, samplesPerBlock, qualityMode);
}

void VerbXPluginProcessor::prepareRealtimeContext(
    double sampleRate,
    int samplesPerBlock,
    int qualityMode
) {
    char error[256] = {};
    verbx_plugin_realtime_config config{};
    config.host_sample_rate = static_cast<unsigned int>(sampleRate);
    config.max_block_frames = static_cast<size_t>(juce::jmax(samplesPerBlock, 1));
    config.channel_count = static_cast<size_t>(juce::jmax(
        getTotalNumInputChannels(),
        getTotalNumOutputChannels()
    ));
    config.quality_mode = static_cast<verbx_plugin_quality_mode>(qualityMode);

    acquireRealtimeContextGuard();
    const auto result = verbx_plugin_realtime_prepare(
        &realtimeContext_,
        &config,
        error,
        sizeof(error)
    );
    if (result != 0) {
        verbx_plugin_realtime_release(&realtimeContext_);
        preparedQualityMode_.store(-1, std::memory_order_release);
        internalSampleRate_.store(0U, std::memory_order_release);
        oversamplingFactor_.store(0U, std::memory_order_release);
    } else {
        preparedQualityMode_.store(qualityMode, std::memory_order_release);
        internalSampleRate_.store(
            verbx_plugin_realtime_internal_sample_rate(&realtimeContext_),
            std::memory_order_release
        );
        oversamplingFactor_.store(
            verbx_plugin_realtime_oversampling_factor(&realtimeContext_),
            std::memory_order_release
        );
    }

    setLatencySamples(static_cast<int>(verbx_plugin_realtime_latency_frames(&realtimeContext_)));
    releaseRealtimeContextGuard();
}

void VerbXPluginProcessor::releaseResources() {
    cancelPendingUpdate();
    acquireRealtimeContextGuard();
    verbx_plugin_realtime_release(&realtimeContext_);
    preparedHostSampleRate_.store(0.0, std::memory_order_release);
    preparedBlockSize_.store(0, std::memory_order_release);
    preparedQualityMode_.store(-1, std::memory_order_release);
    internalSampleRate_.store(0U, std::memory_order_release);
    oversamplingFactor_.store(0U, std::memory_order_release);
    releaseRealtimeContextGuard();
}

bool VerbXPluginProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    const auto& input = layouts.getMainInputChannelSet();
    const auto& output = layouts.getMainOutputChannelSet();
    return input == output
        && (output == juce::AudioChannelSet::mono() || output == juce::AudioChannelSet::stereo());
}

verbx_plugin_realtime_params VerbXPluginProcessor::currentRealtimeParams() const {
    verbx_plugin_realtime_params params{};
    params.pre_delay_ms = parameterValue(parameterPointers_.preDelayMs);
    params.room_size = parameterValue(parameterPointers_.roomSize);
    params.rt60_coarse_normalized = parameterValue(parameterPointers_.rt60Coarse);
    params.rt60_fine_bipolar = parameterValue(parameterPointers_.rt60Fine);
    params.damping = parameterValue(parameterPointers_.damping);
    params.width = parameterValue(parameterPointers_.width);
    params.diffusion = parameterValue(parameterPointers_.diffusion);
    params.wet = parameterValue(parameterPointers_.wet);
    params.dry = parameterValue(parameterPointers_.dry);
    params.freeze = parameterValue(parameterPointers_.freeze) >= 0.5f ? 1 : 0;
    params.reverse = parameterValue(parameterPointers_.reverse) >= 0.5f ? 1 : 0;
    return params;
}

void VerbXPluginProcessor::processBlock(
    juce::AudioBuffer<float>& buffer,
    juce::MidiBuffer& midiMessages
) {
    juce::ignoreUnused(midiMessages);
    juce::ScopedNoDenormals noDenormals;

    constexpr size_t maxChannels = 64U;
    const auto channels = static_cast<size_t>(buffer.getNumChannels());
    const auto frames = static_cast<size_t>(buffer.getNumSamples());
    if (channels == 0U || channels > maxChannels) {
        return;
    }

    const float* inputs[maxChannels] = {};
    float* outputs[maxChannels] = {};
    for (size_t channel = 0; channel < channels; ++channel) {
        inputs[channel] = buffer.getReadPointer(static_cast<int>(channel));
        outputs[channel] = buffer.getWritePointer(static_cast<int>(channel));
    }

    const auto params = currentRealtimeParams();
    if (realtimeContextGuard_.test_and_set(std::memory_order_acquire)) {
        pushAnalyzerSamples(buffer);
        return;
    }
    verbx_plugin_realtime_process(
        &realtimeContext_,
        inputs,
        outputs,
        frames,
        channels,
        &params,
        nullptr
    );
    releaseRealtimeContextGuard();
    pushAnalyzerSamples(buffer);
}

void VerbXPluginProcessor::pushAnalyzerSamples(const juce::AudioBuffer<float>& buffer) noexcept {
    const auto channels = buffer.getNumChannels();
    const auto frames = buffer.getNumSamples();
    if (channels <= 0 || frames <= 0) {
        return;
    }

    const auto write = analyzerWritePosition_.load(std::memory_order_relaxed);
    const auto read = analyzerReadPosition_.load(std::memory_order_acquire);
    const auto occupied = juce::jmin(write - read, analyzerBufferCapacity);
    const auto writable = analyzerBufferCapacity - occupied;
    const auto samplesToWrite = juce::jmin(
        static_cast<std::uint32_t>(frames),
        writable
    );
    const auto channelScale = 1.0f / static_cast<float>(channels);

    for (std::uint32_t sample = 0U; sample < samplesToWrite; ++sample) {
        float mono = 0.0f;
        for (int channel = 0; channel < channels; ++channel) {
            mono += buffer.getReadPointer(channel)[sample];
        }
        analyzerBuffer_[(write + sample) & (analyzerBufferCapacity - 1U)] = mono * channelScale;
    }
    analyzerWritePosition_.store(write + samplesToWrite, std::memory_order_release);
}

int VerbXPluginProcessor::popAnalyzerSamples(float* destination, int maxSamples) noexcept {
    if (destination == nullptr || maxSamples <= 0) {
        return 0;
    }

    const auto read = analyzerReadPosition_.load(std::memory_order_relaxed);
    const auto write = analyzerWritePosition_.load(std::memory_order_acquire);
    const auto available = juce::jmin(write - read, analyzerBufferCapacity);
    const auto samplesToRead = juce::jmin(
        static_cast<std::uint32_t>(maxSamples),
        available
    );
    for (std::uint32_t sample = 0U; sample < samplesToRead; ++sample) {
        destination[sample] = analyzerBuffer_[(read + sample) & (analyzerBufferCapacity - 1U)];
    }
    analyzerReadPosition_.store(read + samplesToRead, std::memory_order_release);
    return static_cast<int>(samplesToRead);
}

double VerbXPluginProcessor::analyzerSampleRate() const noexcept {
    return analyzerSampleRate_.load(std::memory_order_acquire);
}

double VerbXPluginProcessor::effectiveRt60Seconds() const noexcept {
    return verbx_plugin_map_rt60_seconds(
        parameterValue(parameterPointers_.rt60Coarse),
        parameterValue(parameterPointers_.rt60Fine)
    );
}

unsigned int VerbXPluginProcessor::internalSampleRate() const noexcept {
    return internalSampleRate_.load(std::memory_order_acquire);
}

size_t VerbXPluginProcessor::oversamplingFactor() const noexcept {
    return oversamplingFactor_.load(std::memory_order_acquire);
}

int VerbXPluginProcessor::preparedBlockSize() const noexcept {
    return preparedBlockSize_.load(std::memory_order_acquire);
}

void VerbXPluginProcessor::acquireRealtimeContextGuard() noexcept {
    while (realtimeContextGuard_.test_and_set(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

void VerbXPluginProcessor::releaseRealtimeContextGuard() noexcept {
    realtimeContextGuard_.clear(std::memory_order_release);
}

void VerbXPluginProcessor::parameterChanged(const juce::String& changedParameterId, float newValue) {
    if (changedParameterId != parameterId(VERBX_PLUGIN_PARAM_QUALITY_MODE)) {
        return;
    }
    requestedQualityMode_.store(
        juce::jlimit(
            static_cast<int>(VERBX_PLUGIN_QUALITY_HOST),
            static_cast<int>(VERBX_PLUGIN_QUALITY_TARGET_192K),
            juce::roundToInt(newValue)
        ),
        std::memory_order_release
    );
    triggerAsyncUpdate();
}

void VerbXPluginProcessor::handleAsyncUpdate() {
    const auto sampleRate = preparedHostSampleRate_.load(std::memory_order_acquire);
    const auto blockSize = preparedBlockSize_.load(std::memory_order_acquire);
    const auto qualityMode = requestedQualityMode_.load(std::memory_order_acquire);
    if (sampleRate <= 0.0
        || blockSize <= 0
        || qualityMode == preparedQualityMode_.load(std::memory_order_acquire)) {
        return;
    }

    suspendProcessing(true);
    prepareRealtimeContext(sampleRate, blockSize, qualityMode);
    suspendProcessing(false);
    updateHostDisplay(juce::AudioProcessorListener::ChangeDetails().withLatencyChanged(true));
}

juce::AudioProcessorEditor* VerbXPluginProcessor::createEditor() {
    return new VerbXPluginEditor(*this);
}

bool VerbXPluginProcessor::hasEditor() const { return true; }
const juce::String VerbXPluginProcessor::getName() const { return "VERBX"; }
bool VerbXPluginProcessor::acceptsMidi() const { return false; }
bool VerbXPluginProcessor::producesMidi() const { return false; }
bool VerbXPluginProcessor::isMidiEffect() const { return false; }
double VerbXPluginProcessor::getTailLengthSeconds() const { return 360.0; }
int VerbXPluginProcessor::getNumPrograms() { return 1; }
int VerbXPluginProcessor::getCurrentProgram() { return 0; }
void VerbXPluginProcessor::setCurrentProgram(int index) { juce::ignoreUnused(index); }
const juce::String VerbXPluginProcessor::getProgramName(int index) {
    juce::ignoreUnused(index);
    return {};
}
void VerbXPluginProcessor::changeProgramName(int index, const juce::String& newName) {
    juce::ignoreUnused(index, newName);
}

void VerbXPluginProcessor::getStateInformation(juce::MemoryBlock& destData) {
    const auto stateXml = parameters_.copyState().createXml();
    copyXmlToBinary(*stateXml, destData);
}

void VerbXPluginProcessor::setStateInformation(const void* data, int sizeInBytes) {
    const auto stateXml = getXmlFromBinary(data, sizeInBytes);
    if (stateXml != nullptr) {
        const auto restoredState = juce::ValueTree::fromXml(*stateXml);
        if (restoredState.isValid() && restoredState.hasType(parameters_.state.getType())) {
            parameters_.replaceState(restoredState);
        }
    }
}

juce::AudioProcessorValueTreeState& VerbXPluginProcessor::state() {
    return parameters_;
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new VerbXPluginProcessor();
}
