"""
Entry point for the YouTube Partner Estimator mobile application.

This Kivy-based mobile application evaluates YouTube channels as marketing partners
by analyzing videos and viewer comments using BERT sentiment analysis.
"""

from youtube_partner_estimator_mobile import YouTubePartnerApp

if __name__ == '__main__':
    app = YouTubePartnerApp()
    app.run()