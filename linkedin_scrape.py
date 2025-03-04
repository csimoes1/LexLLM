from linkedin_api import Linkedin
import getpass

# Function to authenticate and fetch a profile
def get_linkedin_profile():
    # Get credentials securely
    print("Enter your LinkedIn credentials (used only for authentication):")
    username = input("Email: ")
    password = getpass.getpass("Password: ")  # Hides password input

    # Target profile public ID (e.g., 'williamhgates' for Bill Gates)
    target_profile = input("Enter the target LinkedIn public profile ID (e.g., 'williamhgates'): ")

    try:
        # Authenticate with LinkedIn using your credentials
        api = Linkedin(username, password)

        # Fetch the target profile
        profile = api.get_profile(target_profile)

        # Print key profile details
        print("\nProfile Data:")
        print(f"Full Name: {profile.get('firstName', 'N/A')} {profile.get('lastName', 'N/A')}")
        print(f"Headline: {profile.get('headline', 'N/A')}")
        print(f"Location: {profile.get('locationName', 'N/A')}")
        print(f"Summary: {profile.get('summary', 'N/A')[:200]}...")  # Truncated for brevity
        print("\nExperience:")
        for exp in profile.get('experience', []):
            print(f"- {exp.get('title', 'N/A')} at {exp.get('companyName', 'N/A')} "
                  f"({exp.get('timePeriod', {}).get('startDate', {}).get('year', 'N/A')} - "
                  f"{exp.get('timePeriod', {}).get('endDate', {}).get('year', 'Present')})")

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Possible issues: Invalid credentials, 2FA enabled, or target profile not found.")

# Run the program
if __name__ == "__main__":
    get_linkedin_profile()