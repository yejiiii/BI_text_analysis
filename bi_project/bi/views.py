from django.shortcuts import render
from .models import ReviewWrite, ReviewPredict
from rest_framework.views import View
from django.template.response import TemplateResponse
import predict
# Create your views here.

class WriteReview(View):
    def get(self, request):
        template_name = "bi/write_review.html"
        model = ReviewWrite

        return TemplateResponse(request, template_name)


class PredictReview(View):
    def get(self, request):

        template_name = "bi/predict_overall.html"
        model = ReviewPredict

        return TemplateResponse(request, template_name)

    def post(self, request):
        get_data = request.POST.get("comment","")
        template_name = "bi/predict_overall.html"
        model = ReviewPredict

        predict_score = predict.predict_data(get_data)[0]
        return TemplateResponse(request, template_name, {'score': predict_score})
# def index(request):
#     return render(request, 'bi/write_review.html')
