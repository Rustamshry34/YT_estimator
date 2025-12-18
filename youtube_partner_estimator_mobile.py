"""
Mobile application for evaluating YouTube channels as marketing partners using Kivy.
This app analyzes videos and viewer comments using BERT sentiment analysis
and calculates engagement scores for sponsored and unsponsored videos.
"""

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.popup import Popup
from kivy.uix.spinner import Spinner
from kivy.uix.progressbar import ProgressBar
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.graphics import Color, Rectangle
from kivy.metrics import dp
from kivy.properties import NumericProperty, StringProperty
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.utils import get_color_from_hex
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.animation import Animation
from kivy.uix.widget import Widget
from kivy.graphics.context_instructions import Color
from kivy.graphics.vertex_instructions import Line, RoundedRectangle
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from transformers import pipeline
import time
import os
import regex
import torch
import re
import isodate
import numpy as np
from threading import Thread


# Define Kivy styling
Builder.load_string('''
<ModernButton@Button>:
    canvas.before:
        Color:
            rgba: 0.156, 0.203, 0.741, 1  # #2834BD
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [dp(8)]
    background_normal: ''
    background_down: ''
    color: 1, 1, 1, 1
    font_size: dp(16)
    size_hint_y: None
    height: dp(50)
    bold: True

<ModernTextInput@TextInput>:
    multiline: False
    font_size: dp(16)
    padding: dp(15)
    background_color: 1, 1, 1, 1
    foreground_color: 0, 0, 0, 1
    hint_text_color: 0.6, 0.6, 0.6, 1
    cursor_color: 0.156, 0.203, 0.741, 1  # #2834BD
    canvas.before:
        Color:
            rgba: 0.878, 0.878, 0.878, 1  # #E0E0E0
        Line:
            rectangle: self.x, self.y, self.width, self.height
            width: dp(2)
        Color:
            rgba: 1, 1, 1, 1
        Rectangle:
            pos: self.pos
            size: self.size

<ModernLabel@Label>:
    color: 0.2, 0.2, 0.2, 1
    font_size: dp(16)

<ModernCard@BoxLayout>:
    orientation: 'vertical'
    padding: dp(15)
    spacing: dp(10)
    canvas.before:
        Color:
            rgba: 1, 1, 1, 1
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [dp(12)]
        Color:
            rgba: 0, 0, 0, 0.1
        Line:
            rounded_rectangle: self.x+dp(3), self.y+dp(3), self.width-dp(6), self.height-dp(6), dp(12), 100, 100, 100, 100
''')

class ProgressPopup(Popup):
    """Popup for showing progress during analysis"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = "Processing..."
        self.size_hint = (0.8, 0.4)
        self.auto_dismiss = False
        
        layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(15))
        
        self.progress_bar = ProgressBar(max=100, size_hint_y=None, height=dp(20))
        self.message_label = Label(text="Initializing...", halign='center', text_size=(self.width*0.8, None))
        
        layout.add_widget(self.message_label)
        layout.add_widget(self.progress_bar)
        
        cancel_btn = Button(text='Cancel', size_hint_y=None, height=dp(50))
        cancel_btn.bind(on_press=self.dismiss)
        layout.add_widget(cancel_btn)
        
        self.content = layout
    
    def update_progress(self, message, progress_value):
        self.message_label.text = message
        self.progress_bar.value = progress_value


class YouTubeAnalyzer:
    """Handles YouTube API interactions and analysis"""
    
    def __init__(self):
        self.api_key = 'YOUR_YOUTUBE_API_KEY'  # Replace with actual API key
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        self.sentiment_pipeline = None
        self.setup_sentiment_analysis()
    
    def setup_sentiment_analysis(self):
        """Initialize the BERT sentiment analysis model"""
        try:
            os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
            
            # Initialize multilingual sentiment analysis pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment-finetunned",
                device=0 if torch.cuda.is_available() else -1,
                truncation=True,
                max_length=512
            )
        except Exception as e:
            print(f"Error loading sentiment model: {str(e)}")
            # Fallback to a simpler model
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                    device=0 if torch.cuda.is_available() else -1,
                    truncation=True,
                    max_length=512
                )
            except Exception as e2:
                print(f"Error loading fallback model: {str(e2)}")
    
    def extract_channel_id(self, youtube_url):
        """Extract channel ID from various YouTube URL formats"""
        youtube_url = youtube_url.strip()

        # Handle @username format
        match = re.match(r'(https?://)?(www\.)?youtube\.com/@([a-zA-Z0-9_-]+)', youtube_url)
        if match:
            username = match.group(3)
            try:
                request = self.youtube.search().list(
                    part='snippet',
                    q=username,
                    type='channel',
                    maxResults=1
                ).execute()
        
                if request['items']:
                    return request['items'][0]['snippet']['channelId']
            except Exception as e:
                raise ValueError(f"Could not find channel ID for username: {username}")

        # Handle channel ID in URL
        match = re.search(r'\"externalId\":\"([a-zA-Z0-9_-]+)\"', youtube_url)
        if match:
            return match.group(1)

        # Handle /channel/ format
        match = re.match(r'(https?://)?(www\.)?youtube\.com/channel/([a-zA-Z0-9_-]+)', youtube_url)
        if match:
            return match.group(3)

        # Handle /user/ format
        match = re.match(r'(https?://)?(www\.)?youtube\.com/user/([a-zA-Z0-9_-]+)', youtube_url)
        if match:
            username = match.group(3)
            return self.get_channel_id_by_username(username)

        # Handle /c/ format
        match = re.match(r'(https?://)?(www\.)?youtube\.com/c/([a-zA-Z0-9_-]+)', youtube_url)
        if match:
            custom_name = match.group(3)
            return self.get_channel_id_by_custom_name(custom_name) 

        # Handle general format
        match = re.match(r'(https?://)?(www\.)?youtube\.com/([a-zA-Z0-9_-]+)', youtube_url)
        if match:
            homepage_name = match.group(3)
            return self.get_channel_id_by_custom_name(homepage_name)

        raise ValueError("Invalid YouTube URL format. Please enter a valid channel URL.")

    def get_channel_id_by_username(self, username):
        """Get channel ID by username"""
        request = self.youtube.channels().list(part='id', forUsername=username).execute()
        if request['items']:
            return request['items'][0]['id']
        else:
            raise ValueError(f"Could not find a channel for username: {username}")

    def get_channel_id_by_custom_name(self, custom_name):
        """Get channel ID by custom name"""
        request = self.youtube.search().list(part='snippet', q=custom_name, type='channel', maxResults=1).execute()
        if request['items']:
            return request['items'][0]['snippet']['channelId']
        else:
            raise ValueError(f"Could not find a channel for custom name: {custom_name}")
    
    def get_channel_videos(self, channel_id, video_count):
        """Fetch videos from the channel"""
        videos = []
        next_page_token = None
        base_video_url = "https://www.youtube.com/watch?v="

        while len(videos) < video_count:
            request = self.youtube.search().list(
                part='snippet',
                channelId=channel_id,
                maxResults=min(2, video_count - len(videos)),
                type='video',
                order='date',
                pageToken=next_page_token
            ).execute()

            video_ids = [item['id']['videoId'] for item in request.get('items', []) if 'videoId' in item['id']]
            if not video_ids:
                break

            # Get content details to check for paid promotions
            video_details_request = self.youtube.videos().list(
                part="contentDetails",
                id=",".join(video_ids)
            ).execute()

            for item, video_details in zip(request['items'], video_details_request['items']):
                if 'videoId' in item['id']:
                    video_id = item['id']['videoId']
                    video_title = item['snippet']['title']
                    video_url = f"{base_video_url}{video_id}"

                    duration = video_details['contentDetails']['duration']
                    parsed_duration = isodate.parse_duration(duration).total_seconds()

                    if parsed_duration < 180:  # Skip videos less than 3 minutes
                        continue

                    videos.append({
                        'video_id': video_id, 
                        'video_title': video_title, 
                        'video_url': video_url,
                        'has_paid_promotion': video_details["contentDetails"].get("hasPaidPromotion", False)
                    })

            next_page_token = request.get('nextPageToken')
            if not next_page_token:
                break

        return videos

    def get_video_description(self, video_id):
        """Get video description"""
        request = self.youtube.videos().list(
            part='snippet',
            id=video_id
        ).execute()

        if request['items']:
            return request['items'][0]['snippet']['description']
        else:
            return None

    def get_video_engagement_metrics(self, video_id):
        """Get engagement metrics for a video"""
        try:
            request = self.youtube.videos().list(
                part="statistics",
                id=video_id
            ).execute()

            if 'items' in request and len(request['items']) > 0:
                stats = request['items'][0]['statistics']
                likes = int(stats.get('likeCount', 0))
                views = int(stats.get('viewCount', 0))
                comments_count = int(stats.get('commentCount', 0))
                return likes, views, comments_count
            else:
                return 0, 0, 0
        except Exception as e:
            print(f"Error fetching engagement metrics for video {video_id}: {str(e)}")
            return 0, 0, 0

    def get_comments(self, video_id):
        """Get comments for a video"""
        comments = []
        
        try:
            request = self.youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100
            ).execute()

            for item in request['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

            while 'nextPageToken' in request:
                request = self.youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=100,
                    pageToken=request['nextPageToken']
                ).execute()

                for item in request['items']:
                    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                    comments.append(comment)

            return comments

        except HttpError as e:
            if e.resp.status == 403 and 'commentsDisabled' in str(e):
                return []
            else:
                raise

    def clean_text(self, text):
        """Clean text for sentiment analysis"""
        text = text.lower()
        # Remove URLs
        text = regex.sub(r"http\S+|www\S+", "", text)
        # Remove mentions (@username)
        text = regex.sub(r"@\w+", "", text)
        # Remove hashtags (#example)
        text = regex.sub(r"#\w+", "", text)
        # Remove email addresses
        text = regex.sub(r"\S+@\S+", "", text)
        # Remove numbers (including patterns like "01 09")
        text = regex.sub(r"\b\d+(?:\s+\d+)*\b", "", text)
        # Remove emojis
        text = regex.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)
        # Remove any characters not belonging to letters or whitespace
        text = regex.sub(r"[^\p{L}\s]", "", text, flags=regex.UNICODE)
        # Normalize whitespace
        text = regex.sub(r'\s+', ' ', text)

        return text.strip()

    def analyze_sentiment(self, comments):
        """Analyze sentiment of comments"""
        if not comments or not self.sentiment_pipeline:
            return 0, {'Positive': 0, 'Negative': 0, 'Neutral': 0}

        sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        scores = []

        for comment in comments[:50]:  # Limit to first 50 comments to avoid performance issues
            try:
                cleaned_comment = self.clean_text(comment)
                if not cleaned_comment.strip():
                    continue
                    
                result = self.sentiment_pipeline(cleaned_comment, truncation=True, max_length=512)[0]
                
                # Determine sentiment based on label
                label = result['label'].lower()
                
                # Different models may have different labels
                if any(pos_word in label for pos_word in ['positive', 'pos', '4', '5']):
                    sentiment_counts['Positive'] += 1
                    normalized_score = 1.0
                elif any(neg_word in label for neg_word in ['negative', 'neg', '1', '2']):
                    sentiment_counts['Negative'] += 1
                    normalized_score = -1.0
                else:
                    sentiment_counts['Neutral'] += 1
                    normalized_score = 0.0
                
                scores.append(normalized_score)
            except Exception as e:
                print(f"Error analyzing comment: {str(e)}")
                continue

        mean_sentiment = np.mean(scores) if scores else 0
        return round(mean_sentiment, 3), sentiment_counts

    def is_video_sponsored(self, video_id, video_title, video_description):
        """Check if video is sponsored using YouTube API"""
        # First check if the video has paid promotion via YouTube API
        request = self.youtube.videos().list(
            part="contentDetails",
            id=video_id
        ).execute()
        
        has_paid_promotion = False
        if request['items']:
            has_paid_promotion = request["items"][0]["contentDetails"].get("hasPaidPromotion", False)
        
        # Also check for sponsorship keywords in title/description
        sponsored_keywords = [
            'sponsored', 'paid promotion', 'partnered with', 
            'includes paid promotion', 'brand deal', 'paid partnership',
            'collaboration', 'in collaboration with', 'promoted'
        ]
        combined_text = f"{video_title} {video_description}".lower()
        has_keyword_sponsorship = any(keyword in combined_text for keyword in sponsored_keywords)
        
        return has_paid_promotion or has_keyword_sponsorship

    def get_channel_name(self, channel_id):
        """Get channel name"""
        request = self.youtube.channels().list(part='snippet', id=channel_id).execute()
        if request['items']:
            return request['items'][0]['snippet']['title']
        else:
            return None

    def evaluate_channel(self, channel_id, video_count, progress_callback=None):
        """Main evaluation function"""
        videos = self.get_channel_videos(channel_id, video_count)
        sponsored_sentiments = []
        unsponsored_sentiments = []
        sponsored_sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        unsponsored_sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        sponsored_engagement_metrics = {"likes": 0, "views": 0, "comments": 0, "count": 0}
        unsponsored_engagement_metrics = {"likes": 0, "views": 0, "comments": 0, "count": 0}
        sponsored_videos = []
        unsponsored_videos = []

        for idx, video in enumerate(videos):
            if progress_callback:
                progress_callback(f"Analyzing video {idx+1}/{len(videos)}: {video['video_title']}", int((idx+1)/len(videos)*100))
            
            video_id = video['video_id']
            video_title = video['video_title']
            video_description = self.get_video_description(video_id)

            comments = self.get_comments(video_id)
            if not comments:
                continue

            comments = [self.clean_text(comment) for comment in comments if comment.strip()]
            likes, views, comments_count = self.get_video_engagement_metrics(video_id)

            # Check if video is sponsored
            is_sponsored = self.is_video_sponsored(video_id, video_title, video_description)

            if is_sponsored:
                avg_sentiment, comment_sentiment_counts = self.analyze_sentiment(comments)
                sponsored_sentiments.append(avg_sentiment)

                sponsored_sentiment_counts['Positive'] += comment_sentiment_counts['Positive']
                sponsored_sentiment_counts['Negative'] += comment_sentiment_counts['Negative']
                sponsored_sentiment_counts['Neutral'] += comment_sentiment_counts['Neutral']

                sponsored_engagement_metrics['likes'] += likes
                sponsored_engagement_metrics['views'] += views
                sponsored_engagement_metrics['comments'] += comments_count
                sponsored_engagement_metrics['count'] += 1
                sponsored_videos.append(video)
            else:
                avg_sentiment, comment_sentiment_counts = self.analyze_sentiment(comments)
                unsponsored_sentiments.append(avg_sentiment)

                unsponsored_sentiment_counts['Positive'] += comment_sentiment_counts['Positive']
                unsponsored_sentiment_counts['Negative'] += comment_sentiment_counts['Negative']
                unsponsored_sentiment_counts['Neutral'] += comment_sentiment_counts['Neutral']

                unsponsored_engagement_metrics['likes'] += likes
                unsponsored_engagement_metrics['views'] += views
                unsponsored_engagement_metrics['comments'] += comments_count
                unsponsored_engagement_metrics['count'] += 1
                unsponsored_videos.append(video)

        num_sponsored = len(sponsored_videos)
        num_unsponsored = len(unsponsored_videos)

        avg_sponsored_sentiment = np.mean(sponsored_sentiments) if sponsored_sentiments else 0
        avg_unsponsored_sentiment = np.mean(unsponsored_sentiments) if unsponsored_sentiments else 0

        if sponsored_engagement_metrics['views'] > 0:
            avg_sponsored_engagement_score = (sponsored_engagement_metrics['likes'] + sponsored_engagement_metrics['comments']) / sponsored_engagement_metrics['views']
        else:
            avg_sponsored_engagement_score = 0

        if unsponsored_engagement_metrics['views'] > 0:
            avg_unsponsored_engagement_score = (unsponsored_engagement_metrics['likes'] + unsponsored_engagement_metrics['comments']) / unsponsored_engagement_metrics['views']
        else:
            avg_unsponsored_engagement_score = 0

        overall_sponsored_score = (0.7 * avg_sponsored_sentiment) + (0.3 * avg_sponsored_engagement_score)
        overall_unsponsored_score = (0.7 * avg_unsponsored_sentiment) + (0.3 * avg_unsponsored_engagement_score)

        return {
            'channel_id': channel_id,
            'sponsored_score': overall_sponsored_score,
            'unsponsored_score': overall_unsponsored_score,
            'num_sponsored': num_sponsored,
            'num_unsponsored': num_unsponsored,
            'Sponsored_Sentiment_Score': avg_sponsored_sentiment,
            'Unsponsored_Sentiment_Score': avg_unsponsored_sentiment,
            'Sponsored_Engagement_Score': avg_sponsored_engagement_score,
            'Unsponsored_Engagement_Score': avg_unsponsored_engagement_score,
            'sponsored_positive_comments': sponsored_sentiment_counts['Positive'],
            'sponsored_negative_comments': sponsored_sentiment_counts['Negative'],
            'sponsored_neutral_comments': sponsored_sentiment_counts['Neutral'],
            'unsponsored_positive_comments': unsponsored_sentiment_counts['Positive'],
            'unsponsored_negative_comments': unsponsored_sentiment_counts['Negative'],
            'unsponsored_neutral_comments': unsponsored_sentiment_counts['Neutral']
        }


class MainScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Main layout
        main_layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(15))
        
        # Header
        header_layout = BoxLayout(size_hint_y=None, height=dp(120), padding=dp(10))
        with header_layout.canvas.before:
            Color(rgba=get_color_from_hex('#2834BD'))
            Rectangle(pos=header_layout.pos, size=header_layout.size)
        
        header_label = Label(
            text='YouTube Partner Estimator',
            font_size=dp(24),
            color=(1, 1, 1, 1),
            bold=True
        )
        subtitle_label = Label(
            text='Analyze YouTube channels for partnership potential',
            font_size=dp(16),
            color=(0.9, 0.9, 0.9, 1)
        )
        
        header_content = BoxLayout(orientation='vertical', padding=dp(10))
        header_content.add_widget(header_label)
        header_content.add_widget(subtitle_label)
        
        header_layout.add_widget(header_content)
        main_layout.add_widget(header_layout)
        
        # Input section
        input_card = BoxLayout(orientation='vertical', size_hint_y=None, height=dp(150), padding=dp(15), spacing=dp(10))
        with input_card.canvas.before:
            Color(rgba=(1, 1, 1, 1))
            RoundedRectangle(pos=input_card.pos, size=input_card.size, radius=[dp(12)])
        
        # Channel URL input
        url_label = Label(text='Channel URL:', size_hint_y=None, height=dp(30), halign='left')
        self.url_input = ModernTextInput(hint_text='Enter YouTube Channel URL')
        
        # Video count and evaluate button
        bottom_input_layout = BoxLayout(size_hint_y=None, height=dp(60), spacing=dp(10))
        
        count_label = Label(text='Videos to analyze:', size_hint_x=0.4)
        self.video_count_spinner = Spinner(
            text='4',
            values=['2', '4', '6', '8', '10', '12', '15', '20'],
            size_hint_x=0.3
        )
        
        self.evaluate_button = ModernButton(text='Evaluate Channel')
        self.evaluate_button.bind(on_press=self.start_evaluation)
        
        bottom_input_layout.add_widget(count_label)
        bottom_input_layout.add_widget(self.video_count_spinner)
        bottom_input_layout.add_widget(self.evaluate_button)
        
        input_card.add_widget(url_label)
        input_card.add_widget(self.url_input)
        input_card.add_widget(bottom_input_layout)
        main_layout.add_widget(input_card)
        
        # Results section
        results_card = BoxLayout(orientation='vertical', padding=dp(15), spacing=dp(10))
        with results_card.canvas.before:
            Color(rgba=(1, 1, 1, 1))
            RoundedRectangle(pos=results_card.pos, size=results_card.size, radius=[dp(12)])
        
        results_header = Label(
            text='Analysis Results',
            font_size=dp(20),
            color=get_color_from_hex('#2834BD'),
            size_hint_y=None,
            height=dp(40)
        )
        
        # Scrollable results area
        self.results_scroll = ScrollView(size_hint_y=1)
        self.results_grid = GridLayout(cols=2, spacing=dp(5), size_hint_y=None)
        self.results_grid.bind(minimum_height=self.results_grid.setter('height'))
        
        self.results_scroll.add_widget(self.results_grid)
        
        results_card.add_widget(results_header)
        results_card.add_widget(self.results_scroll)
        main_layout.add_widget(results_card)
        
        # Recommendation section
        recommendation_card = BoxLayout(orientation='vertical', size_hint_y=None, height=dp(100), padding=dp(15))
        with recommendation_card.canvas.before:
            Color(rgba=(1, 1, 1, 1))
            RoundedRectangle(pos=recommendation_card.pos, size=recommendation_card.size, radius=[dp(12)])
        
        recommendation_header = Label(
            text='Partnership Recommendation',
            font_size=dp(18),
            color=get_color_from_hex('#2834BD'),
            size_hint_y=None,
            height=dp(30)
        )
        
        self.recommendation_label = Label(
            text='Enter a YouTube channel URL and click \'Evaluate Channel\' to get a recommendation',
            font_size=dp(16),
            text_size=(None, None),
            halign='center',
            valign='middle'
        )
        
        recommendation_card.add_widget(recommendation_header)
        recommendation_card.add_widget(self.recommendation_label)
        main_layout.add_widget(recommendation_card)
        
        self.add_widget(main_layout)
        
        # Initialize analyzer
        self.analyzer = YouTubeAnalyzer()
        self.progress_popup = None

    def start_evaluation(self, instance):
        """Start the channel evaluation process"""
        channel_url = self.url_input.text.strip()
        if not channel_url:
            self.show_error("Please enter a YouTube channel URL")
            return

        try:
            video_count = int(self.video_count_spinner.text)
        except ValueError:
            video_count = 4  # default value

        # Show progress popup
        self.progress_popup = ProgressPopup()
        self.progress_popup.open()

        # Start evaluation in a separate thread
        self.eval_thread = Thread(
            target=self.run_evaluation,
            args=(channel_url, video_count)
        )
        self.eval_thread.daemon = True
        self.eval_thread.start()

    def run_evaluation(self, channel_url, video_count):
        """Run the evaluation in a separate thread"""
        try:
            def update_progress(message, progress):
                if self.progress_popup:
                    Clock.schedule_once(lambda dt: self.progress_popup.update_progress(message, progress))

            channel_id = self.analyzer.extract_channel_id(channel_url)
            channel_name = self.analyzer.get_channel_name(channel_id)
            
            update_progress("Evaluating channel...", 10)
            
            results = self.analyzer.evaluate_channel(
                channel_id, 
                video_count, 
                progress_callback=update_progress
            )
            
            results['channel_name'] = channel_name
            
            # Schedule UI updates on main thread
            Clock.schedule_once(lambda dt: self.handle_results(results))
            
        except Exception as e:
            Clock.schedule_once(lambda dt: self.handle_error(str(e)))

    def handle_results(self, results):
        """Handle the results from the evaluation"""
        if self.progress_popup:
            self.progress_popup.dismiss()
            self.progress_popup = None

        # Clear previous results
        self.results_grid.clear_widgets()
        
        # Group metrics for better organization
        metrics_groups = {
            "Channel Information": [
                ('Channel Name', results['channel_name']),
                ('Channel ID', results['channel_id'])
            ],
            "Performance Scores": [
                ('Sponsored Content Score', f"{results['sponsored_score']:.4f}"),
                ('Organic Content Score', f"{results['unsponsored_score']:.4f}")
            ],
            "Video Analysis": [
                ('Sponsored Videos Analyzed', str(results['num_sponsored'])),
                ('Organic Videos Analyzed', str(results['num_unsponsored']))
            ],
            "Sentiment Analysis": [
                ('Sponsored Sentiment Score', f"{results['Sponsored_Sentiment_Score']:.4f}"),
                ('Organic Sentiment Score', f"{results['Unsponsored_Sentiment_Score']:.4f}")
            ],
            "Engagement Metrics": [
                ('Sponsored Engagement Score', f"{results['Sponsored_Engagement_Score']:.4f}"),
                ('Organic Engagement Score', f"{results['Unsponsored_Engagement_Score']:.4f}")
            ],
            "Comment Analysis (Sponsored)": [
                ('Positive Comments', str(results['sponsored_positive_comments'])),
                ('Negative Comments', str(results['sponsored_negative_comments'])),
                ('Neutral Comments', str(results['sponsored_neutral_comments']))
            ],
            "Comment Analysis (Organic)": [
                ('Positive Comments', str(results['unsponsored_positive_comments'])),
                ('Negative Comments', str(results['unsponsored_negative_comments'])),
                ('Neutral Comments', str(results['unsponsored_neutral_comments']))
            ]
        }
        
        # Add grouped metrics to the grid
        for group, metrics in metrics_groups.items():
            # Add group header
            group_header = Label(
                text=f"--- {group} ---",
                font_size=dp(14),
                bold=True,
                color=get_color_from_hex('#2834BD'),
                size_hint_y=None,
                height=dp(30),
                halign='left'
            )
            blank_label = Label(text='', size_hint_y=None, height=dp(30))
            
            self.results_grid.add_widget(group_header)
            self.results_grid.add_widget(blank_label)
            
            # Add metrics
            for metric, value in metrics:
                metric_label = Label(
                    text=metric,
                    font_size=dp(14),
                    size_hint_y=None,
                    height=dp(30),
                    halign='left'
                )
                value_label = Label(
                    text=str(value),
                    font_size=dp(14),
                    size_hint_y=None,
                    height=dp(30),
                    halign='left'
                )
                
                self.results_grid.add_widget(metric_label)
                self.results_grid.add_widget(value_label)

        # Update recommendation
        if results['sponsored_score'] > results['unsponsored_score']:
            self.recommendation_label.text = "✨ This channel shows strong potential for sponsored partnerships! The sponsored content performs well with positive audience engagement and sentiment."
            self.recommendation_label.color = get_color_from_hex('#2E7D32')
            with self.recommendation_label.canvas.before:
                Color(rgba=get_color_from_hex('#E8F5E9'))
                Rectangle(pos=self.recommendation_label.pos, size=self.recommendation_label.size)
        else:
            self.recommendation_label.text = "⚠️ This channel may need improvement before pursuing sponsored partnerships. The organic content currently outperforms sponsored content in terms of engagement and sentiment."
            self.recommendation_label.color = get_color_from_hex('#F57C00')
            with self.recommendation_label.canvas.before:
                Color(rgba=get_color_from_hex('#FFF8E1'))
                Rectangle(pos=self.recommendation_label.pos, size=self.recommendation_label.size)

    def handle_error(self, error_message):
        """Handle errors during evaluation"""
        if self.progress_popup:
            self.progress_popup.dismiss()
            self.progress_popup = None
        self.show_error(error_message)

    def show_error(self, message):
        """Show error message in a popup"""
        popup = Popup(
            title='Error',
            content=Label(text=message),
            size_hint=(0.8, 0.4)
        )
        popup.open()


class YouTubePartnerApp(App):
    def build(self):
        # Set window size for mobile simulation
        Window.size = (400, 700)
        
        sm = ScreenManager()
        sm.add_widget(MainScreen(name='main'))
        
        return sm

    def on_start(self):
        """Called when the app starts"""
        pass


if __name__ == '__main__':
    YouTubePartnerApp().run()