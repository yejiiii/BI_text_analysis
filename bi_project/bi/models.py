from django.db import models
from django.utils import timezone

# Create your models here.

class ReviewWrite(models.Model):
    review_data = models.TextField()

    def __str__(self):
        return self.review_data

class ReviewPredict(models.Model):
    review_predict = models.TextField()

    def __str__(self):
        return self.review_predict
