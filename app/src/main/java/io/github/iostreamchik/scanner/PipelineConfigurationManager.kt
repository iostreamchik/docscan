package io.github.iostreamchik.scanner

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/**
 * Manages shared pipeline configuration state.
 * Provides a centralized way for ViewModels to observe and update pipeline type.
 */
class PipelineConfigurationManager {
    private val _pipelineType = MutableStateFlow(PipelineType.ADAPTIVE)
    val pipelineType: StateFlow<PipelineType> = _pipelineType

    fun setPipelineType(type: PipelineType) {
        _pipelineType.value = type
    }
}