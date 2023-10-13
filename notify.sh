# define email addresses in the config file
config_file="notify_settings"

if [ -f ${config_file} ]; then
	source ${config_file}
	date | mail -s "done" -a "From: ${from_addr}" "${to_addr}"
fi
