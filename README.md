# YouTube Partner Estimator - Mobile Application

This is a Kivy-based mobile application that evaluates the potential of YouTube channels as marketing partners by analyzing videos and viewer comments. It utilizes BERT to classify comments as positive, neutral, or negative and calculates engagement scores for sponsored and unsponsored videos.

## Features

- Mobile-friendly interface built with Kivy
- Analyzes YouTube channels for partnership potential
- Uses YouTube Data API to detect sponsored content (no scraping)
- Performs sentiment analysis on comments using BERT
- Calculates engagement scores for both sponsored and organic content
- Provides actionable insights and recommendations
- Modern UI with progress indicators and organized results

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Obtain a YouTube Data API key from the [Google Cloud Console](https://console.cloud.google.com/)

3. Replace `'YOUR_YOUTUBE_API_KEY'` in the `YouTubeAnalyzer` class with your actual API key

## Usage

Run the application:
```bash
python youtube_partner_estimator_mobile.py
```

## Key Changes from PyQt Version

1. **No Selenium**: Removed all Selenium-based web scraping
2. **YouTube Data API**: Now uses `hasPaidPromotion` field from the YouTube API to detect sponsored content
3. **Mobile-Optimized UI**: Built with Kivy for mobile compatibility
4. **Modern Styling**: Includes custom styling for buttons, inputs, and cards
5. **Progress Tracking**: Shows detailed progress during analysis
6. **Responsive Design**: Adapts to different screen sizes

## Architecture

- `YouTubePartnerApp`: Main Kivy application class
- `MainScreen`: Primary screen with all UI elements
- `YouTubeAnalyzer`: Handles all YouTube API interactions and analysis
- `ProgressPopup`: Shows progress during evaluation
- Various custom Kivy widgets for modern UI components

## API Usage

The application uses the following YouTube Data API endpoints:
- `search().list()`: To get channel videos
- `videos().list()`: To get video details including `hasPaidPromotion` status
- `commentThreads().list()`: To get video comments
- `channels().list()`: To get channel information

## Sentiment Analysis

Uses a multilingual BERT model for sentiment analysis:
- Primary model: `cardiffnlp/twitter-xlm-roberta-base-sentiment-finetunned`
- Fallback model: `nlptown/bert-base-multilingual-uncased-sentiment`

## Evaluation Metrics

The application calculates:
- Sponsored content performance score
- Organic content performance score
- Sentiment analysis scores
- Engagement metrics
- Comment sentiment distribution
- Partnership recommendations