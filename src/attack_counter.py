# Script to calculate the total, successful, and failed attacks with percentages
def calculate_attack_statistics(file_path):
    total_attacks = 0
    success_count = 0
    failed_count = 0

    # Open and read the file
    with open(file_path, 'r') as file:
        for line in file:
            # Check if the line contains the result of an attack
            if '- SUCCESS' in line:
                success_count += 1
            elif '- FAILED' in line:
                failed_count += 1

            # Every line containing '- SUCCESS' or '- FAILED' is an attack
            if '- SUCCESS' in line or '- FAILED' in line:
                total_attacks += 1

    # Calculate the percentages
    success_percentage = (success_count / total_attacks) * 100 if total_attacks > 0 else 0
    failed_percentage = (failed_count / total_attacks) * 100 if total_attacks > 0 else 0

    # Print the results
    print(f"Total attacks: {total_attacks}")
    print(f"Successful attacks: {success_count} ({success_percentage:.2f}%)")
    print(f"Failed attacks: {failed_count} ({failed_percentage:.2f}%)")

# Call the function with the path to your output.txt file
calculate_attack_statistics('output.txt')
