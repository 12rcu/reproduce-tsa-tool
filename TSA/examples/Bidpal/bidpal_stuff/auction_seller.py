import time
from interact_bidpal import BidPalUser
bin_dir = "../challenge_program/bin"

def main():
    AUCTION_ID = "auction1"
    user0 = BidPalUser(bin_dir, "8000", "user0")
    user2 = BidPalUser(bin_dir, "8002", "user2")
    print("Users set up.")

    time.sleep(1)
    user0.connect("nuc2", "8001")
    user2.connect("nuc2", "8001")
    user0.connect("localhost", "8002")
    print("Users connected.")

    user0.start_auction("Applied Cryptography book, new condition", "auction1")

    # give users a chance to receive and process auction start message
    user2.wait_for_output("received auction start announcement")
    time.sleep(1) # give users time to process the start announcement

    print("Starting bidding.")
    # two users bid 
    user2.bid("auction1", "500")
    user2.wait_for_output("received a bid commitment")
    
    # wait for comparisons to happen
    print(user2.wait_for_compare_results)

    user0.end_auction("auction1") # announce end of bidding
    print("Ending auction1")

    user0.wait_for_outputs(["auction concession", "received a win claim"]) #can't guarantee what order they'll come in

    user0.get_auction_status("auction1") # check status of auction1

    user0.get_bidders("auction1") # determine who's in the running to win

    user0.announce_winner("auction1", "user2", 500) # announce winner
    #user1.wait_for_output("received end of auction")
    time.sleep(1)
    # print all users' view of auction1 outcome
    #user1.get_auction_status("auction1")

    # quit
    user0.quit()

if __name__=="__main__":
    main()
