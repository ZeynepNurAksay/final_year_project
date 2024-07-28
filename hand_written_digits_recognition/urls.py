from django.urls import path

from . import views

# app_name = "final_project"

urlpatterns = [
    # path for index page
    path("", views.index, name="index"),
    # image upload links
    path("european", views.european, name="european"),
    path("chinese", views.chinese, name="chinese"),
    path("arabic", views.arabic, name="arabic"),
    # pdf page upload links
    path("european_page/<int:page_id>", views.european_page, name="european_page"),
    path("chinese_page/<int:page_id>", views.chinese_page, name="chinese_page"),
    path("arabic_page/<int:page_id>", views.arabic_page, name="arabic_page")
]
