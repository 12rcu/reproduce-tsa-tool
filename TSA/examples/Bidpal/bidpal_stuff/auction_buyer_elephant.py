import time
from interact_bidpal import BidPalUser
bin_dir = "../challenge_program/bin"

def main():
    user1 = BidPalUser(bin_dir, "8001", "user1")
    print("User set up.")
    time.sleep(1)

    user1.wait_for_output("Connected")
    user1.wait_for_output("Connected")
    print("User connected.")

    # give users a chance to receive and process auction start message
    user1.wait_for_output("received auction start announcement")
    time.sleep(1) # give users time to process the start announcement

    # two users bid
    user1.wait_for_output("received a bid commitment")
    user1.bid("auction1", "elephant")

    # wait for comparisons to happen
    print(user1.wait_for_compare_results()) # user1's bid compared to user0's

    user1.get_auction_status("auction1")    

    user1.wait_for_output("received end of auction")
    
    # print all users' view of auction1 outcome
    user1.get_auction_status("auction1")

    # quit
    user1.quit()

if __name__=="__main__":
    main()
