# Generated by Django 4.2.5 on 2023-09-20 09:39

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0013_alter_student_fav_subject_alter_student_weak_sub'),
    ]

    operations = [
        migrations.CreateModel(
            name='TestScore',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('q_5x5', models.CharField(max_length=5)),
                ('q_vowels', models.CharField(max_length=5)),
                ('q_powerhouse', models.CharField(max_length=100)),
                ('q_unit_of_force', models.CharField(max_length=100)),
                ('q_chemical_symbol', models.CharField(max_length=5)),
                ('q_father_of_nation', models.CharField(max_length=100)),
                ('q_founder_of_computer', models.CharField(max_length=100)),
                ('q_music_notation', models.CharField(max_length=100)),
                ('q_contour_line', models.CharField(max_length=100)),
                ('q_world_war_year', models.CharField(max_length=5)),
                ('score', models.IntegerField()),
                ('score_category', models.CharField(max_length=50)),
                ('student', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='accounts.student')),
            ],
        ),
    ]
