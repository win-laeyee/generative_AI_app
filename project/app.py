from flask import Flask, render_template, request
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

app = Flask(__name__)
qa_history = [] # Initialize a list to store the question-answer history

# Load the fine-tuned BlenderBot model and tokenizer from the path to my best model
model_name = 'best_model' 
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

@app.route('/')
def index():
    """Render the main page with a question input form and QA history."""
    return render_template('question_form.html', qa_history=qa_history)

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle user's question, generate a model response, and update QA history."""
    if request.method == 'POST':
        user_question = request.form.get('question')

        # Tokenize the user's question
        inputs = tokenizer(user_question, return_tensors="pt", padding=True, truncation=True)

        # Generate model response
        response = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"])

        # Decode the response
        generated_response = tokenizer.decode(response[0], skip_special_tokens=True)

        # Append the question and generated answer to the QA history
        qa_history.append({'question': user_question, 'answer': generated_response})

        return render_template('response.html', response=generated_response)
     
     # Render the question input form
    return render_template('question_form.html')

if __name__ == '__main__':
    app.run(debug=True) # Run the Flask app in debug mode if this script is the main entry point
