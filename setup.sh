mkdir -p ~/.streamlit/

echo "[theme]
primaryColor='#7792E3'
backgroundColor='#FFFFFF'
secondaryBackgroundColor='#F0F2F6'
textColor='#262730'
font='sans serif'
[server]
headless = true
enableCORS=false
port = $PORT
" > ~/.streamlit/config.toml