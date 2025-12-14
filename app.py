"""
Text Summarization Web Application
NLP Project - College Assignment
Author: [Your Name]
Date: [Current Date]
Course: Natural Language Processing
"""

from flask import Flask, request, render_template
from transformers import pipeline
import torch
import os
import sys

app = Flask(__name__)

# ========== LOAD MODELS ==========
# Note: We'll try to load a fine-tuned model first, otherwise use the base model
summarizer = None
model_loaded = False

print("=== Starting up Text Summarization App ===")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

# Check if GPU is available
if torch.cuda.is_available():
    print("✅ GPU found! Will use GPU for faster processing")
    device = 0  # Use GPU
else:
    print("⚠️  No GPU found, using CPU (might be slower)")
    device = -1  # Use CPU

try:
    # First try to load our fine-tuned model
    model_path = "./models/bart-dialogsum"
    
    if os.path.exists(model_path):
        print(f"📂 Found our trained model at: {model_path}")
        try:
            # Load our fine-tuned model
            summarizer = pipeline(
                "summarization",
                model=model_path,
                tokenizer=model_path,
                device=device
            )
            print("✅ Our trained model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading our model: {e}")
            print("🔄 Trying to load the base model instead...")
            # Fall back to base BART model
            summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=device
            )
    else:
        print("📦 No trained model found, loading base BART model...")
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=device
        )
    
    model_loaded = True
    print("✨ Model is ready to use!")
    
except Exception as e:
    print(f"❌ BIG ERROR loading model: {e}")
    model_loaded = False
    summarizer = None

# ========== HELPER FUNCTIONS ==========

def clean_text(txt):
    """
    Clean the input text by removing extra whitespace and normalizing
    This helps the model work better
    """
    if not txt:
        return ""
    
    # Remove extra whitespace
    import re
    txt = re.sub(r'\s+', ' ', txt)
    txt = txt.strip()
    
    return txt

def count_words(text):
    """Simple function to count words in text"""
    return len(text.split())

def calculate_stats(input_text, summary_text):
    """Calculate statistics about the summarization"""
    input_words = count_words(input_text)
    summary_words = count_words(summary_text)
    
    if summary_words > 0:
        compression_ratio = round(input_words / summary_words, 2)
    else:
        compression_ratio = 0
    
    return {
        'input_words': input_words,
        'summary_words': summary_words,
        'compression_ratio': compression_ratio,
        'input_chars': len(input_text),
        'summary_chars': len(summary_text)
    }

def generate_summary(text, max_length=150, min_length=40):
    """
    Generate a summary of the input text
    This is the main function that uses the AI model
    """
    if not model_loaded or summarizer is None:
        return "Error: Model not loaded. Please check if the model files are available."
    
    if not text or len(text.strip()) < 10:
        return "Error: Text too short. Please enter at least 10 characters."
    
    try:
        # Clean the text first
        clean_text_input = clean_text(text)
        
        # Set up parameters for the model
        # These are default parameters that work well for summarization
        params = {
            'max_length': max_length,      # Maximum length of summary
            'min_length': min_length,      # Minimum length of summary
            'length_penalty': 2.0,         # Encourages longer summaries
            'num_beams': 4,                # Beam search for better quality
            'early_stopping': True,        # Stop when good enough
            'no_repeat_ngram_size': 3      # Avoid repeating phrases
        }
        
        print(f"📝 Generating summary for text ({len(clean_text_input)} chars)...")
        
        # Generate the summary
        result = summarizer(clean_text_input, **params)
        
        # Extract the summary text from the result
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and 'summary_text' in result[0]:
                summary = result[0]['summary_text']
            else:
                summary = str(result[0])
        else:
            summary = "Could not generate summary. Please try again."
        
        print(f"✅ Summary generated ({len(summary)} chars)")
        return summary
        
    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        return f"Error: {str(e)}"

# ========== SAMPLE DIALOGUES ==========
# These are example texts that users can try
SAMPLE_DIALOGUES = [
    {
        'id': 1,
        'title': 'Doctor Appointment',
        'category': 'Healthcare',
        'text': """#Person1#: Hi, Mr. Smith. I'm Doctor Hawkins. Why are you here today?
#Person2#: I found it would be a good idea to get a check-up.
#Person1#: Yes, well, you haven't had one for 5 years. You should have one every year.
#Person2#: I know. I figure as long as there is nothing wrong, why go see the doctor?
#Person1#: Well, the best way to avoid serious illnesses is to find out about them early. So try to come at least once a year for your own good.
#Person2#: Ok.
#Person1#: Let me see here. Your eyes and ears look fine. Take a deep breath, please. Do you smoke, Mr. Smith?"""
    },
    {
        'id': 2,
        'title': 'Job Interview',
        'category': 'Employment',
        'text': """#Person1#: Tell me about your previous work experience.
#Person2#: I worked as a software developer at TechCorp for three years. I was responsible for developing web applications.
#Person1#: What technologies did you use?
#Person2#: I primarily used Python, Django, and React. I also worked with Docker and AWS.
#Person1#: Why are you interested in this position?
#Person2#: I'm looking for new challenges and your company's focus on AI aligns with my interests."""
    },
    {
        'id': 3,
        'title': 'Travel Planning',
        'category': 'Travel',
        'text': """#Person1#: Have you decided where we should go for our vacation?
#Person2#: I was thinking about Japan. What do you think?
#Person1#: Japan sounds amazing! When were you thinking of going?
#Person2#: Maybe in the spring to see the cherry blossoms. What's your budget?
#Person1#: I can spend around $3000 for the whole trip.
#Person2#: That should be enough for a 10-day trip if we plan carefully."""
    }
]

# ========== FLASK ROUTES ==========

@app.route('/')
def home():
    """
    Main page - shows the text summarization interface
    Similar to your resume.html but for text summarization
    """
    # Get model status for display
    status = "✅ Model Loaded Successfully" if model_loaded else "❌ Model Not Available"
    
    return render_template(
        "summarize.html",
        model_status=status,
        model_loaded=model_loaded,
        sample_dialogues=SAMPLE_DIALOGUES
    )

@app.route('/summarize', methods=['POST'])
def summarize():
    """
    Handle the summarization request
    Similar to your /pred route but for text summarization
    """
    if not model_loaded:
        return render_template(
            "summarize.html",
            error="Model not loaded. Please check if the model files are available.",
            sample_dialogues=SAMPLE_DIALOGUES
        )
    
    # Get the text from the form
    text = request.form.get('text', '').strip()
    
    # Get parameters (with defaults)
    try:
        max_length = int(request.form.get('max_length', 150))
        min_length = int(request.form.get('min_length', 40))
    except:
        max_length = 150
        min_length = 40
    
    # Validate input
    if not text:
        return render_template(
            "summarize.html",
            error="Please enter some text to summarize.",
            sample_dialogues=SAMPLE_DIALOGUES
        )
    
    if len(text) < 10:
        return render_template(
            "summarize.html",
            error="Text too short. Please enter at least 10 characters.",
            sample_dialogues=SAMPLE_DIALOGUES,
            input_text=text
        )
    
    if len(text) > 10000:
        return render_template(
            "summarize.html",
            error="Text too long. Maximum 10,000 characters allowed.",
            sample_dialogues=SAMPLE_DIALOGUES,
            input_text=text[:1000] + "... [truncated]"
        )
    
    # Generate summary
    print(f"📊 Processing request: {len(text)} chars, max_len={max_length}, min_len={min_length}")
    
    summary = generate_summary(text, max_length, min_length)
    
    # Calculate statistics
    stats = calculate_stats(text, summary)
    
    # Get model info
    model_info = "Fine-tuned BART model" if os.path.exists("./models/bart-dialogsum") else "Base BART model"
    
    return render_template(
        "summarize.html",
        input_text=text,
        summary=summary,
        stats=stats,
        max_length=max_length,
        min_length=min_length,
        model_info=model_info,
        sample_dialogues=SAMPLE_DIALOGUES,
        success=True
    )

@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    """
    API endpoint for programmatic access
    Returns JSON instead of HTML
    """
    if not model_loaded:
        return {
            'success': False,
            'error': 'Model not loaded',
            'summary': ''
        }
    
    # Get data from JSON request
    if request.is_json:
        data = request.get_json()
        text = data.get('text', '').strip()
        max_length = data.get('max_length', 150)
        min_length = data.get('min_length', 40)
    else:
        # Get from form data
        text = request.form.get('text', '').strip()
        max_length = request.form.get('max_length', 150)
        min_length = request.form.get('min_length', 40)
    
    # Validate
    if not text:
        return {
            'success': False,
            'error': 'No text provided',
            'summary': ''
        }
    
    # Generate summary
    summary = generate_summary(text, max_length, min_length)
    
    # Check if there was an error
    if summary.startswith('Error:'):
        return {
            'success': False,
            'error': summary,
            'summary': ''
        }
    
    # Calculate stats
    stats = calculate_stats(text, summary)
    
    return {
        'success': True,
        'summary': summary,
        'statistics': stats,
        'parameters': {
            'max_length': max_length,
            'min_length': min_length
        }
    }

@app.route('/api/health')
def api_health():
    """API health check endpoint"""
    return {
        'status': 'healthy' if model_loaded else 'error',
        'model_loaded': model_loaded,
        'gpu_available': torch.cuda.is_available(),
        'timestamp': '2024'  # You could use datetime here
    }

@app.route('/api/examples')
def api_examples():
    """Get example dialogues"""
    return {
        'success': True,
        'examples': SAMPLE_DIALOGUES,
        'count': len(SAMPLE_DIALOGUES)
    }

# ========== MAIN ==========

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Text Summarization Web App Starting...")
    print("="*50)
    print(f"🌐 Web interface: http://localhost:5000")
    print(f"🔧 API endpoint: http://localhost:5000/api/summarize")
    print(f"🏥 Health check: http://localhost:5000/api/health")
    print("="*50 + "\n")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
