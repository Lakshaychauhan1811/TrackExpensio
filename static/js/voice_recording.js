// Voice Recording Functions for TrackExpensio
// This module handles speech-to-text functionality using Whisper

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

async function toggleVoiceRecording() {
    if (!ensureAuthenticated()) {
        showMessage('Please sync with Google before using voice commands', 'warning');
        return;
    }
    
    const micButton = document.getElementById('chatMicButton');
    const voiceStatus = document.getElementById('voiceStatus');
    const voiceStatusText = document.getElementById('voiceStatusText');
    
    if (!isRecording) {
        // Start recording
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            audioChunks = [];
            
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };
            
            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                await processVoiceCommand(audioBlob);
                
                // Stop all tracks
                stream.getTracks().forEach(track => track.stop());
            };
            
            mediaRecorder.start();
            isRecording = true;
            
            // Update UI
            micButton.classList.add('recording');
            voiceStatus.style.display = 'flex';
            voiceStatus.className = 'voice-status';
            voiceStatusText.textContent = 'Listening... Speak now';
            
            // Auto-stop after 30 seconds
            setTimeout(() => {
                if (isRecording) {
                    stopRecording();
                }
            }, 30000);
            
        } catch (error) {
            console.error('Error accessing microphone:', error);
            showMessage('Microphone access denied. Please allow microphone permissions.', 'error');
        }
    } else {
        // Stop recording
        stopRecording();
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        
        const micButton = document.getElementById('chatMicButton');
        const voiceStatus = document.getElementById('voiceStatus');
        const voiceStatusText = document.getElementById('voiceStatusText');
        
        micButton.classList.remove('recording');
        voiceStatus.className = 'voice-status processing';
        voiceStatusText.textContent = 'Processing your voice...';
    }
}

async function processVoiceCommand(audioBlob) {
    const auth = getAuthPayload();
    if (!auth) {
        showMessage('Session required', 'warning');
        return;
    }
    
    const voiceStatus = document.getElementById('voiceStatus');
    const voiceStatusText = document.getElementById('voiceStatusText');
    
    try {
        // Create form data
        const formData = new FormData();
        formData.append('audio', audioBlob, 'voice-command.webm');
        if (auth.api_key) formData.append('api_key', auth.api_key);
        if (auth.session_id) formData.append('session_id', auth.session_id);
        
        // Send to backend
        const response = await fetch('/api/speech-to-text', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok || data.status === 'error') {
            throw new Error(data.message || 'Voice processing failed');
        }
        
        // Show transcribed text
        const transcribedText = data.transcribed_text;
        const reply = data.reply || 'Command processed';
        
        // Update UI
        voiceStatus.className = 'voice-status success';
        voiceStatusText.textContent = `Heard: "${transcribedText}"`;
        
        // Add transcribed message to chat
        addChatMessage(`🎤 ${transcribedText}`, 'user');
        chatHistory.push({ role: 'user', content: transcribedText });
        
        // Add bot response
        addChatMessage(reply, 'bot');
        chatHistory.push({ role: 'assistant', content: reply });
        
        // Refresh data if needed
        if (data.tool_results && data.tool_results.length > 0) {
            refreshDataForTools(data.tool_results);
        }
        
        // Hide status after 3 seconds
        setTimeout(() => {
            voiceStatus.style.display = 'none';
        }, 3000);
        
        showMessage('✅ Voice command processed successfully!', 'success');
        
    } catch (error) {
        console.error('Error processing voice command:', error);
        
        voiceStatus.className = 'voice-status error';
        voiceStatusText.textContent = `Error: ${error.message}`;
        
        setTimeout(() => {
            voiceStatus.style.display = 'none';
        }, 3000);
        
        showMessage('Failed to process voice command', 'error');
    }
}

// Check for microphone support
function checkMicrophoneSupport() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        const micButton = document.getElementById('chatMicButton');
        if (micButton) {
            micButton.style.display = 'none';
        }
        console.warn('Microphone API not supported in this browser');
    }
}

// Initialize microphone check on load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', checkMicrophoneSupport);
} else {
    checkMicrophoneSupport();
}
