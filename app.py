from flask import Flask, render_template, request, jsonify
from huggingface_hub import HfApi, login
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
from collections import Counter
from gensim import corpora
from gensim.models import LdaModel
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Login to Hugging Face
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
if huggingface_token:
    login(huggingface_token)
else:
    print("Warning: HUGGINGFACE_TOKEN not found in environment variables")

class AdvancedChatbotManager:
    def __init__(self, model, tokenizer, history_file: str = "chat_history.json"):
        self.model = model
        self.repo_name = request.form['repo_name']
        self.tokenizer = tokenizer
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.backup_folder = "ch_backup"
        self.history_file = history_file
        self.max_history = 15
        self.max_repetition_threshold = 0.7
        self.min_response_length = 15
        self.max_response_length = 150
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.topic_model = self._initialize_topic_model()
        self.user_feedback = []

    def _initialize_topic_model(self):
        dictionary = corpora.Dictionary([["topic", "model", "initialization"]])
        corpus = [dictionary.doc2bow(["topic", "model", "initialization"])]
        return LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=1)

    def load_chat_history(self) -> List[Dict]:
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
                if self._validate_history(history):
                    return history
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.warning(f"Error loading chat history: {e}")
        return []

    def _validate_history(self, history: List[Dict]) -> bool:
        if not isinstance(history, list):
            return False
        for entry in history:
            if not isinstance(entry, dict) or 'human' not in entry or 'ai' not in entry or 'timestamp' not in entry:
                return False
            if not isinstance(entry['human'], str) or not isinstance(entry['ai'], str) or not isinstance(entry['timestamp'], str):
                return False
        return True

    def save_chat_history(self, history: List[Dict]) -> None:
        try:
            
            if not os.path.exists(self.backup_folder):
                os.makedirs(self.backup_folder)
            
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            backup_file = os.path.join(self.backup_folder, f"chat_history_{timestamp}.json")
            
            # Create a backup of the existing history file
            if os.path.exists(self.history_file):
                os.replace(self.history_file, backup_file)

            # Save the new chat history
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving chat history: {e}")

    def detect_repetition(self, response: str, history: List[Dict]) -> bool:
        if not history:
            return False

        recent_responses = [entry['ai'] for entry in history[-5:]]
        for past_response in recent_responses:
            similarity = self._calculate_similarity(response, past_response)
            if similarity > self.max_repetition_threshold:
                return True
        return False

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def check_response_quality(self, response: str, user_input: str) -> bool:
        if len(response.split()) < self.min_response_length:
            return False
        if len(response.split()) > self.max_response_length:
            return False
        if response.count('.') > 15:
            return False
        if len(set(response.split())) < len(response.split()) * 0.4:
            return False
        if not self._check_relevance(user_input, response):
            return False
        return True

    # def _check_relevance(self, response: str, user_input: str) -> bool:
    #     words1 = set(user_input.lower().split())
    #     words2 = set(response.lower().split())
    #
    #     intersection = len(words1.intersection(words2))
    #     min_overlap = 1 if len(words1) < 3 else 2
    #
    #     return intersection >= min_overlap

    def _check_relevance(self, user_query, chatbot_response, threshold=0.55):
        embeddings = HuggingFaceEmbeddings()

        query_embedding = embeddings.embed_query(user_query)
        response_embedding = embeddings.embed_query(chatbot_response)
        similarity = cosine_similarity(
            [query_embedding], [response_embedding]
        )[0][0]
        return similarity >= threshold

    def generate_response(self, user_input: str, history: List[Dict], max_attempts: int = 5) -> str:
        history_text = self._format_history(history)
        # input_text = f"{history_text}\nHuman: {user_input}\nAI:"
        for attempt in range(max_attempts):
            try:

                hf = HuggingFacePipeline.from_model_id(
                    model_id=self.repo_name,
                    task="text-generation",
                    pipeline_kwargs={
                        "max_new_tokens": 150,
                        "num_return_sequences": 1,
                        "no_repeat_ngram_size": 3,
                        "top_k": 50,
                        "top_p": 0.92,
                        "temperature": self._dynamic_temperature(attempt),
                        "do_sample":True
                    },
                )
                template = "{history_text}\nHuman: {user_input}\nAI:"
                prompt = PromptTemplate.from_template(template)
                chain = prompt | hf.bind(skip_prompt=True)
                input_data = {
                    "history_text": history_text,
                    "user_input": user_input,
                }

                # inputs = self.tokenizer.encode_plus(
                #     input_text,
                #     return_tensors="pt",
                #     padding='max_length',
                #     max_length=512,
                #     truncation=True
                # ).to(self.device)
                #
                # with torch.no_grad():
                #     outputs = self.model.generate(
                #         inputs['input_ids'],
                #         attention_mask=inputs['attention_mask'],
                #         max_new_tokens=150,
                #         num_return_sequences=1,
                #         no_repeat_ngram_size=3,
                #         top_k=50,
                #         top_p=0.92,
                #         temperature=self._dynamic_temperature(attempt),
                #         do_sample=True
                #     )
                #
                # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # response = response.split("AI:")[-1].strip()
                response = chain.invoke(input_data).strip()

                if (self.check_response_quality(response, user_input) and
                    not self.detect_repetition(response, history)):
                    return response

            except Exception as e:
                logging.error(f"Error generating response (attempt {attempt+1}): {e}")

        return self._generate_fallback_response(user_input, history)

    def _dynamic_temperature(self, attempt: int) -> float:
        return 0.7 + (attempt * 0.05)

    def _generate_fallback_response(self, user_input: str, history: List[Dict]) -> str:
        fallback_responses = [
            "I apologize, but I'm having trouble understanding. Could you please rephrase your message? It could be that it's not part of my training data.",
            "I'm not sure I have enough information to answer that. Can you provide more context? It could be that it's not part of my training data.",
            "That's an interesting question. To better assist you, could you clarify what specific aspect you're most interested in? I think it could be that it's not part of my training data.",
            "I want to make sure I give you the most accurate information. Could you break down your message into smaller parts? It could be that it's not part of my training data.",
            "I'm still learning and evolving. Could you try putting your message in a different way? It could be that it's not part of my training data."
        ]
        return np.random.choice(fallback_responses)

    def _format_history(self, history: List[Dict]) -> str:
        formatted = []
        for entry in history[-self.max_history:]:
            formatted.extend([
                f"Human: {entry['human']}",
                f"AI: {entry['ai']}"
            ])
        return "\n".join(formatted)

    def reset_history(self) -> List[Dict]:
        return []

    def analyze_conversation(self, history: List[Dict]) -> Dict:
        human_words = Counter()
        ai_words = Counter()
        for entry in history:
            human_words.update(entry['human'].lower().split())
            ai_words.update(entry['ai'].lower().split())

        all_text = " ".join([entry['human'] + " " + entry['ai'] for entry in history])
        bow_corpus = self.topic_model.id2word.doc2bow(all_text.lower().split())
        topics = self.topic_model[bow_corpus]

        return {
            'total_exchanges': len(history),
            'top_human_words': human_words.most_common(5),
            'top_ai_words': ai_words.most_common(5),
            'average_human_length': sum(len(entry['human'].split()) for entry in history) / len(history),
            'average_ai_length': sum(len(entry['ai'].split()) for entry in history) / len(history),
            'main_topics': sorted(topics, key=lambda x: x[1], reverse=True)[:3],
        }

    def handle_special_commands(self, user_input: str, history: List[Dict]) -> Tuple[bool, str]:
        if user_input.lower() in ['reset', 'clear']:
            return True, "Conversation history has been reset. How can I help you?"
        elif user_input.lower() in ['analyze', 'stats']:
            analysis = self.analyze_conversation(history)
            return True, f"Conversation Analysis:\n{json.dumps(analysis, indent=2)}"
        elif user_input.lower() in ['help', 'commands']:
            return True, "Available commands: reset/clear, analyze/stats, help/commands, feedback"
        elif user_input.lower() == 'feedback':
            return True, self._get_user_feedback()
        return False, ""

    def _get_user_feedback(self) -> str:
        feedback = input("Please rate the chatbot's performance (1-5, 5 being best): ")
        comment = input("Any additional comments? ")
        self.user_feedback.append({"rating": feedback, "comment": comment, "timestamp": datetime.now().isoformat()})
        return "Thank you for your feedback!"

    def detect_user_intent(self, user_input: str) -> str:
        intents = {
            "greeting": ["hello", "hi", "hey", "greetings"],
            "farewell": ["bye", "goodbye", "see you", "farewell"],
            "question": ["what", "why", "how", "when", "where", "who"],
            "command": ["do", "please", "can you", "could you"],
            "opinion": ["think", "believe", "feel", "opinion"],
        }

        user_input_lower = user_input.lower()
        for intent, keywords in intents.items():
            if any(keyword in user_input_lower for keyword in keywords):
                return intent
        return "general"

# Global variables
chatbot = None
chat_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_commits', methods=['POST'])
def get_commits():
    repo_name = request.form['repo_name']
    api = HfApi()

    try:
        # Get commit history
        commits = api.list_repo_commits(repo_name)
        commit_info = [{"hash": commit.commit_id, "title": commit.title} for commit in commits]
        return jsonify({"commits": commit_info})
    except Exception as e:
        logger.error(f"Error getting commits: {str(e)}")
        return jsonify({"error": str(e)})

@app.route('/load_model', methods=['POST'])
def load_model():
    global chatbot, chat_history

    try:
        repo_name = request.form['repo_name']
        commit_hash = request.form['commit_hash']

        logger.info(f"Loading model from {repo_name} at commit {commit_hash}")

        # Load the model and tokenizer with specific commit hash
        model = AutoModelForCausalLM.from_pretrained(
            repo_name,
            token=huggingface_token,
            trust_remote_code=True,
            revision=commit_hash
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            repo_name,
            token=huggingface_token,
            trust_remote_code=True,
            revision=commit_hash
        )
        
        chatbot = AdvancedChatbotManager(model=model, tokenizer=tokenizer)
        chat_history = chatbot.load_chat_history()

        logger.info("Model loaded successfully")
        return jsonify({"success": True, "message": "Model loaded successfully"})
    except Exception as e:
        logger.error(f"Error in load_model: {str(e)}")
        return jsonify({
            "error": str(e),
            "details": "Please ensure you have proper access to the repository and have set the HUGGINGFACE_TOKEN environment variable."
        })

@app.route('/chat', methods=['POST'])
def chat():
    global chatbot, chat_history

    if not chatbot:
        return jsonify({"error": "Model not loaded. Please choose a model first."})

    user_input = request.form['user_input']

    if user_input.lower() in ['quit', 'exit', 'bye']:
        chatbot.save_chat_history(chat_history)
        return jsonify({"response": "Goodbye! It was a pleasure assisting you.", "end_chat": True})

    special_command, special_response = chatbot.handle_special_commands(user_input, chat_history)
    if special_command:
        if user_input.lower() in ['reset', 'clear']:
            chat_history = chatbot.reset_history()
        return jsonify({"response": special_response})

    if len(user_input) < 2:
        return jsonify({"response": "Could you please provide more details or ask a specific question?"})

    intent = chatbot.detect_user_intent(user_input)

    try:
        response = chatbot.generate_response(user_input, chat_history)

        chat_history.append({
            "human": user_input,
            "ai": response,
            "timestamp": datetime.now().isoformat(),
            "intent": intent
        })

        if len(chat_history) > chatbot.max_history:
            chat_history = chat_history[-chatbot.max_history:]

        chatbot.save_chat_history(chat_history)

        return jsonify({"response": response, "intent": intent})
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({"error": str(e)})

@app.route('/reset_chat', methods=['POST'])
def reset_chat():
    global chatbot, chat_history
    
    try:
        if chatbot:
            chat_history = chatbot.reset_history()
            # Add a welcome message to the chat history
            welcome_message = "Chat history has been reset. How can I help you?"
            chat_history.append({
                "human": "",
                "ai": welcome_message,
                "timestamp": datetime.now().isoformat(),
                "intent": "greeting"
            })
            return jsonify({"success": True, "message": welcome_message})
        else:
            return jsonify({"error": "No active chat session"})
    except Exception as e:
        logger.error(f"Error in reset_chat: {str(e)}")
        return jsonify({"error": str(e)})

@app.route('/delete_message', methods=['POST'])
def delete_message():
    global chat_history
    
    try:
        message_id = request.form.get('message_id')
        if not message_id:
            return jsonify({"error": "No message ID provided"})
            
        # Convert message_id to integer (assuming it's a number)
        msg_idx = int(message_id.split('-')[1]) - 1
        
        if 0 <= msg_idx < len(chat_history):
            # Remove the message and all subsequent messages
            chat_history = chat_history[:msg_idx]
            chatbot.save_chat_history(chat_history)
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Message not found"})
    except Exception as e:
        logger.error(f"Error in delete_message: {str(e)}")
        return jsonify({"error": str(e)})

@app.route('/regenerate_response', methods=['POST'])
def regenerate_response():
    global chatbot, chat_history

    try:
        message_id = request.form.get('message_id')
        user_input = request.form.get('user_input')

        if not message_id or not user_input:
            return jsonify({"error": "Missing message ID or user input"})

        msg_idx = int(message_id.split('-')[1]) - 1

        if 0 <= msg_idx < len(chat_history):
            # Remove all messages after the selected message
            chat_history = chat_history[:msg_idx + 1]
            
            # Regenerate the response
            new_response = chatbot.generate_response(user_input, chat_history)
            
            # Update the chat history
            chat_history.append({
                "human": user_input,
                "ai": new_response,
                "timestamp": datetime.now().isoformat(),
                "intent": chatbot.detect_user_intent(user_input)
            })
            
            chatbot.save_chat_history(chat_history)
            
            return jsonify({"success": True, "response": new_response})
        else:
            return jsonify({"error": "Message not found"})
    except Exception as e:
        logger.error(f"Error in regenerate_response: {str(e)}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
