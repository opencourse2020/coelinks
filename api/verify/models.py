from django.db import models
from .formatChecker import ContentTypeRestrictedFileField


def document_directory_path(instance, filename):

    return 'documents/user_{0}/{1}'.format(instance.user, filename)


def headshot_directory_path(instance, filename):

    return 'headshots/user_{0}/{1}'.format(instance.user, filename)


# Create your models here.
class Document(models.Model):
    user = models.IntegerField()
    file = ContentTypeRestrictedFileField(upload_to=document_directory_path,
                                             content_types=['image/bmp', 'image/gif', 'image/jpeg', 'image/png', ],
                                             max_upload_size=52428800, blank=True, null=True)
    # document1 = ContentTypeRestrictedFileField(upload_to=document_directory_path,
    #                                       content_types=['image/bmp', 'image/gif', 'image/jpeg', 'image/png', ],
    #                                       max_upload_size=52428800, blank=True, null=True)
    # document2 = ContentTypeRestrictedFileField(upload_to=document_directory_path,
    #                                       content_types=['image/bmp', 'image/gif', 'image/jpeg', 'image/png', ],
    #                                       max_upload_size=52428800, blank=True, null=True)
    keyword = models.CharField(max_length=150, null=True, blank=True)

    name = models.CharField(max_length=150, null=True, blank=True)
    name_checked = models.BooleanField(default=False)

    dob = models.CharField(max_length=50, null=True, blank=True)
    dob_checked = models.BooleanField(default=False)

    grade = models.CharField(max_length=10, null=True, blank=True)
    grade_checked = models.BooleanField(default=False)

    uploaded = models.DateTimeField(auto_now_add=True)

    verified = models.BooleanField(default=False)
    scanned = models.BooleanField(default=False)

    def __str__(self):
        return str(self.user)


class HeadShot(models.Model):
    user = models.IntegerField()
    file = ContentTypeRestrictedFileField(upload_to=headshot_directory_path,
                                          content_types=['image/bmp', 'image/gif', 'image/jpeg', 'image/png', ],
                                          max_upload_size=52428800, blank=True, null=True)
    uploaded = models.DateTimeField(auto_now_add=True)
    verified = models.BooleanField(default=False)

    def __str__(self):
        return str(self.user)


class AIModel(models.Model):
    name = models.CharField(max_length=50, null=True, blank=True)
    file = models.FileField(upload_to="aimodels")
    description = models.CharField(max_length=250, null=True, blank=True)
    uploaded = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.name)
