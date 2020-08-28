from django.shortcuts import render
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.contrib.auth.models import User
from django.shortcuts import redirect
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import authenticate, login as auth_login

def custom_login(request):
    if request.user.is_authenticated:
        return redirect('/administrator/')
    else:
        if request.method == 'POST':
            form = AuthenticationForm(request.POST)
            username = request.POST['username']
            password = request.POST['password']
            user = authenticate(username=username, password=password)

            if user is not None:
                if user.is_active:
                    auth_login(request, user)
                    return redirect('/administrator/')
            else:
                messages.error(request,'Username atau Password tidak sesuai!')
                return redirect('/login/')
        else:
            form = AuthenticationForm()
            return render(request, 'registration/login.html', {'form': form})
