cd /bigdata
echo "Start"

python3 src/update_recent_papers.py
echo "update_recent_papers.py done"

python3 src/update_all_papers.py
echo "update_all_papers.py done"

python3 src/update_model.py
echo "update_model.py done"

python3 src/update_front_end_data.py
echo "update_front_end_data.py done"
echo "All done"
