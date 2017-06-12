"""bi_project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.10/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url, include
from django.contrib import admin
from django.views.generic import TemplateView
from bi.models import *
from bi.views import *

urlpatterns = [
    # url(r'', include('bi.urls')),
    url(r'^admin/', admin.site.urls),
    # url(r'^write/', TemplateView.as_view(template_name = 'write_reivew.html')),
    # url(r'^predict/', TemplateView.as_view(template_name = 'predict_overall.html')),
    url(r'^$', WriteReview.as_view(), name='write'),
    url(r'^predict/$', PredictReview.as_view(), name='predict'),
]
