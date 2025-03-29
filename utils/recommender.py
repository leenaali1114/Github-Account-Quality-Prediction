import os
import json
from groq import Groq

class GitHubRecommender:
    def __init__(self, api_key=None):
        """Initialize the GitHub recommender with Groq API."""
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass it directly.")
        
        self.client = Groq(api_key=self.api_key)
    
    def get_recommendations(self, quality_level, user_metrics):
        """Get personalized recommendations based on quality level and user metrics."""
        # Create a prompt for the Groq API
        prompt = f"""
        You are a GitHub expert advisor. Based on the following GitHub account metrics and quality assessment, 
        provide specific, actionable recommendations to improve the account's quality and visibility.
        
        Current quality level: {quality_level}
        
        Account metrics:
        {json.dumps(user_metrics, indent=2)}
        
        Please provide:
        1. An overall assessment of the account's strengths and weaknesses
        2. 5 specific, actionable recommendations to improve the account quality
        3. For each recommendation, explain why it's important and how it will help
        
        Format your response as JSON with the following structure:
        {{
            "assessment": "Overall assessment text here",
            "recommendations": [
                {{
                    "title": "Recommendation title",
                    "description": "Detailed explanation",
                    "importance": "Why this matters",
                    "action_items": ["Step 1", "Step 2", "Step 3"]
                }},
                // More recommendations...
            ]
        }}
        """
        
        # Call Groq API
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a GitHub expert advisor providing actionable recommendations."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-70b-8192",  # Using Llama 3 70B model
                max_tokens=2048,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            recommendations = json.loads(response.choices[0].message.content)
            return recommendations
        
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            # Fallback recommendations if API call fails
            return {
                "assessment": "We couldn't generate personalized recommendations at this time.",
                "recommendations": [
                    {
                        "title": "Create comprehensive README files",
                        "description": "Add detailed README files to all your repositories explaining what they do, how to use them, and how to contribute.",
                        "importance": "Good documentation makes your projects more accessible and useful to others.",
                        "action_items": ["Create a template README", "Add installation instructions", "Include usage examples"]
                    },
                    {
                        "title": "Contribute to open source projects",
                        "description": "Find projects that interest you and start contributing with bug fixes or small features.",
                        "importance": "Shows collaboration skills and increases your visibility in the community.",
                        "action_items": ["Find beginner-friendly issues", "Submit quality pull requests", "Engage in project discussions"]
                    }
                ]
            } 