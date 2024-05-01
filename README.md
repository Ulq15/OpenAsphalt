# OpenAsphalt

- [OpenAsphalt](#openasphalt)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Steps To Run The App](#steps-to-run-the-app)
    - [Steps To Run The Model](#steps-to-run-the-model)
    - [Special Comments](#special-comments)
  - [Main Features](#main-features)
  - [Using the Parking Marketplace: Renters](#using-the-parking-marketplace-renters)
  - [Listing your Parking Spot: Homeowners](#listing-your-parking-spot-homeowners)
  - [Withdraw Funds: Homeowners](#withdraw-funds-homeowners)
  - [Troubleshooting](#troubleshooting)
  - [Customer Support](#customer-support)

## Getting Started

Welcome to a streamlined parking experience. This section will guide you through the basics of setting up and starting to use OpenAsphalt.

### Prerequisites

- Install [Node.js](https://nodejs.org/en/learn/getting-started/how-to-install-nodejs)
  - [Download](https://nodejs.org/en/download)
- Install [Yarn](https://classic.yarnpkg.com/lang/en/docs/install)
- Install [Visual Studio Code](https://code.visualstudio.com/docs/setup/setup-overview)
  - [Download](https://code.visualstudio.com/download)
- Install [Python](https://www.python.org/downloads/)

### Steps To Run The App

Follow these simple steps to view OpenAsphalt on your device. To view and run the GitHub repository code using Visual Studio Code and then execute it using the command `yarn run dev`, you can follow these step-by-step instructions:

1. Clone the Repository:
    First, you need to clone the repository from GitHub to your local machine. You can do this by opening a terminal or command prompt and using the git command:

    ``` shell
    git clone https://github.com/Ulq15/OpenAsphalt.git
    ```

2. Open the Repository in VSCode:
    Once the repository is cloned, open VSCode. Go to File > Open Folder... and select the directory where you cloned the repository.

3. Install Dependencies:
    Before you can run the project, you need to install its dependencies. Make sure you have Node.js and Yarn installed on your computer. Open the integrated terminal in VSCode by going to Terminal > New Terminal. In the terminal, run the following command to install dependencies:

    ```shell
    yarn install
    ```

4. Run the Project:
    After installing all the dependencies, you can start the project by running:

    ```shell
    yarn run dev
    ```

### Steps To Run The Model

### Special Comments

The webapp is user-friendly and intuitive, designed to simplify your parking ex-
perience on event days. All functionalities are straightforward, and setup is quick
and easy!

## Main Features

Discover the core functionalities of OpenAsphalt, including:

- Comprehensive Parking Marketplace: Book driveways as parking spots near stadiums.
- Real-Time Location Services: Use Google Maps to find and navigate to your reserved spot.
- Secure Payment Processing: Complete transactions securely through Stripe.
- User Verification and Check-in System: Verify and manage your bookings efficiently.
- Dispute Resolution Support: Access support for any issues with bookings or payments.

## Using the Parking Marketplace: Renters

This section provides detailed instructions on how to use the parking marketplace:

1. Open the app and use View Listings or View Map to find available parking near Sofi Stadium and Kia Forum.
2. Select a parking spot and view details such as price, location, and events.
3. Book your spot and receive confirmation along with parking, arrival/departure and check-in instructions

## Listing your Parking Spot: Homeowners

This section provides detailed instructions on how to list your parking spot:

1. Open the app and click Create Listing on the Navigation Bar or on List View.
2. Enter your address using Google Maps AutoComplete.
3. Choose Your Events you would like to make your spot available for.
   1. Enter the price and number of spots available for parking
4. Enter a Description, Instructions, Arrival/Departure Time, Phone Number.
5. Upload Listing Images
6. Listing will be made available pending approval within 24 hours.

## Withdraw Funds: Homeowners

This section provides detailed instructions on how to withdraw funds:

1. Open the app and use the Navigation Bar to select ”Payouts”.
2. View Funds available to withdraw and Pending Funds.
   1. Pending Funds will be made available once the event/booking starts
3. Please select payout method and enter how much you would like to cash out. To add/change, please select + Add Method.
4. Click Transfer. Payouts will be processed within 24 hours.

## Troubleshooting

Common issues and quick fixes:

- Booking Cancellation: Please visit the support page to send us an email regarding any cancellations for your booking. Bookings within 24 hours of the events are not refundable.
- Payment not going through: Ensure your payment information is up to date and retry.
- Navigation issues: Verify location services are enabled on your device.

## Customer Support

If you encounter any issues or have questions, contact us at `support@openasphalt.com`.
