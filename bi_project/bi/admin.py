from django.contrib import admin
from .models import ReviewWrite

# Register your models here.

class ReviewWriteAdmin(admin.ModelAdmin):
    list_diplay = ('review_data',)

admin.site.register(ReviewWrite, ReviewWriteAdmin)
